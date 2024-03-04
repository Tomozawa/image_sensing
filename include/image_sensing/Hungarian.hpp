#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <rcpputils/asserts.hpp>

#include <vector>
#include <optional>
#include <variant>

#include <utils.hpp>

namespace hungarian{
    class hungarian final{
        private:
        enum class MaxNumZeroRow : size_t{};
        enum class MaxNumZeroCol : size_t{};

        static std::vector<cv::Point> find_zeros(const cv::Mat matrix){
            rcpputils::require_true(matrix.type() == CV_16S);

            std::vector<cv::Point> result{};
            for(int r = 0; r < matrix.rows; r++){
                for(int c = 0; c < matrix.cols; c++){
                    if(matrix.at<int16_t>(r, c) == 0) result.push_back(cv::Point(c, r));
                }
            }

            return result;
        }

        static void get_lines(const cv::Mat matrix, std::vector<int>& row_lines, std::vector<int>& col_lines){
            std::vector<cv::Point> zeros = find_zeros(matrix);

            auto get_max_num_zero = [](const std::vector<size_t>& r, const std::vector<size_t>& c)->std::variant<MaxNumZeroRow, MaxNumZeroCol>{
                size_t max_num_zero_r = 0;
                size_t max_num_zero_c = 0;
                size_t index_r, index_c;

                for(size_t r_cnt = 0U; r_cnt < r.size(); r_cnt++){
                    if(max_num_zero_r <= r.at(r_cnt)){
                        max_num_zero_r = r.at(r_cnt);
                        index_r = r_cnt;
                    }
                }

                for(size_t c_cnt = 0U; c_cnt < c.size(); c_cnt++){
                    if(max_num_zero_c <= c.at(c_cnt)){
                        max_num_zero_c = c.at(c_cnt);
                        index_c = c_cnt;
                    }
                }

                if(max_num_zero_r >= max_num_zero_c) return static_cast<MaxNumZeroRow>(index_r);
                else return static_cast<MaxNumZeroCol>(index_c);
            };
            
            while(zeros.size() > 0U){
                std::vector<size_t> num_zero_row(matrix.rows, 0U);
                std::vector<size_t> num_zero_col(matrix.cols, 0U);

                for(const auto& zero : zeros){
                    num_zero_row.at(zero.y)++;
                    num_zero_col.at(zero.x)++;
                }

                auto max_num_zero = get_max_num_zero(num_zero_row, num_zero_col);
                int max_num_zero_index;
                auto zeros_ite = zeros.begin();

                switch(max_num_zero.index()){
                    case 0:
                    max_num_zero_index = static_cast<int>(std::get<MaxNumZeroRow>(max_num_zero));
                    row_lines.push_back(static_cast<int>(max_num_zero_index));
                    while(zeros_ite != zeros.end()){
                        if((*zeros_ite).y == max_num_zero_index) zeros_ite = zeros.erase(zeros_ite);
                        else zeros_ite++;
                    }
                    break;

                    default:
                    max_num_zero_index = static_cast<int>(std::get<MaxNumZeroCol>(max_num_zero));
                    col_lines.push_back(max_num_zero_index);
                    while(zeros_ite != zeros.end()){
                        if((*zeros_ite).x == max_num_zero_index)zeros_ite = zeros.erase(zeros_ite);
                        else zeros_ite++;
                    }
                    break;
                }
            }
        }

        static cv::Mat calc_subtract_matrix(const cv::Mat matrix){
            rcpputils::require_true(matrix.type() == CV_16S);

            cv::Mat result = cv::Mat::ones(cv::Size(matrix.cols, matrix.rows), CV_16S);
            cv::Mat min_max_loc_mask(cv::Size(matrix.cols, matrix.rows), CV_8U, UINT8_MAX);

            std::vector<int> row_lines, col_lines;
            get_lines(matrix, row_lines, col_lines);

            for(const auto row_line : row_lines){
                for(int c = 0; c < matrix.cols; c++){
                    result.at<int16_t>(row_line, c)--;
                    min_max_loc_mask.at<uint8_t>(row_line, c) = 0U;
                }
            }

            for(const auto col_line : col_lines){
                for(int r = 0; r < matrix.rows; r++){
                    result.at<int16_t>(r, col_line)--;
                    min_max_loc_mask.at<uint8_t>(r, col_line) = 0U;
                }
            }

            double min_val;
            cv::minMaxLoc(matrix, &min_val, nullptr, nullptr, nullptr, min_max_loc_mask);
            result *= min_val;
            return result;
        }

        static std::vector<cv::Point> do_assign(cv::Mat matrix, const bool transposition){
            rcpputils::require_true(matrix.type() == CV_16S);

            auto try_assign = [transposition](cv::Mat matrix)->std::optional<std::vector<cv::Point>>{
                std::vector<std::optional<int>> zero_rows(matrix.rows, std::nullopt);

                for(int c = 0; c < matrix.cols; c++){
                    for(const auto& zero : find_zeros(matrix.col(c)))
                    if(!zero_rows.at(zero.y).has_value()){
                        zero_rows.at(zero.y) = c;
                        break;
                    }
                }

                std::vector<cv::Point> result{};
                for(int r = 0; r < matrix.rows; r++){
                    if(!zero_rows.at(r).has_value()) return {};
                    result.push_back(
                        (!transposition)? cv::Point(*zero_rows.at(r), r)
                        : cv::Point(r, *zero_rows.at(r))
                    );
                }

                return result;
            };

            std::optional<std::vector<cv::Point>> assignment = try_assign(matrix);

            while(!assignment.has_value()){
                matrix -= calc_subtract_matrix(matrix);
                assignment = try_assign(matrix);
            }

            return *assignment;
        }

        static cv::Mat pre_process(cv::Mat matrix){
            rcpputils::require_true(matrix.type() == CV_16S);

            for(int r = 0; r < matrix.rows; r++){
                double min_val;
                cv::minMaxLoc(matrix.row(r), &min_val);
                matrix.row(r).forEach<int16_t>([min_val](int16_t& val, const int*){val -= min_val;});
            }

            for(int c = 0; c < matrix.cols; c++){
                double min_val;
                cv::minMaxLoc(matrix.col(c), &min_val);
                matrix.col(c).forEach<int16_t>([min_val](int16_t& val, const int*){val -= min_val;});
            }

            return matrix;
        }

        public:
        hungarian() = delete;
        hungarian(const hungarian&) = delete;

        static std::vector<cv::Point> assign(cv::InputArray input, bool transposition = false){
            rcpputils::require_true(input.isMat());
            rcpputils::require_true(input.type() == CV_16S);
            rcpputils::require_true(input.rows() <= input.cols());

            std::vector<cv::Point> result{};
            cv::Mat input_matrix = input.getMat();
            input_matrix.resize(input_matrix.cols);

            pre_process(input_matrix);

            return do_assign(input_matrix, transposition);
        }
    };
}//hungarian