#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <rcpputils/asserts.hpp>

#include <vector>
#include <optional>
#include <variant>
#include <utility>

#include <utils.hpp>

#include <iostream>

namespace hungarian{
    // class hungarian final{
    //     private:
    //     enum class MaxNumZeroRow : size_t{};
    //     enum class MaxNumZeroCol : size_t{};

    //     static std::vector<cv::Point> find_zeros(const cv::Mat matrix){
    //         rcpputils::require_true(matrix.type() == CV_16S);

    //         std::vector<cv::Point> result{};
    //         for(int r = 0; r < matrix.rows; r++){
    //             for(int c = 0; c < matrix.cols; c++){
    //                 if(matrix.at<int16_t>(r, c) == 0) result.push_back(cv::Point(c, r));
    //             }
    //         }

    //         return result;
    //     }

    //     static void get_lines(const cv::Mat matrix, std::vector<int>& row_lines, std::vector<int>& col_lines){
    //         std::vector<cv::Point> zeros = find_zeros(matrix);

    //         auto get_max_num_zero = [](const std::vector<size_t>& r, const std::vector<size_t>& c)->std::variant<MaxNumZeroRow, MaxNumZeroCol>{
    //             size_t max_num_zero_r = 0;
    //             size_t max_num_zero_c = 0;
    //             size_t index_r, index_c;

    //             for(size_t r_cnt = 0U; r_cnt < r.size(); r_cnt++){
    //                 if(max_num_zero_r <= r.at(r_cnt)){
    //                     max_num_zero_r = r.at(r_cnt);
    //                     index_r = r_cnt;
    //                 }
    //             }

    //             for(size_t c_cnt = 0U; c_cnt < c.size(); c_cnt++){
    //                 if(max_num_zero_c <= c.at(c_cnt)){
    //                     max_num_zero_c = c.at(c_cnt);
    //                     index_c = c_cnt;
    //                 }
    //             }

    //             if(max_num_zero_r >= max_num_zero_c) return static_cast<MaxNumZeroRow>(index_r);
    //             else return static_cast<MaxNumZeroCol>(index_c);
    //         };
            
    //         while(zeros.size() > 0U){
    //             std::vector<size_t> num_zero_row(matrix.rows, 0U);
    //             std::vector<size_t> num_zero_col(matrix.cols, 0U);

    //             for(const auto& zero : zeros){
    //                 num_zero_row.at(zero.y)++;
    //                 num_zero_col.at(zero.x)++;
    //             }

    //             auto max_num_zero = get_max_num_zero(num_zero_row, num_zero_col);
    //             int max_num_zero_index;
    //             auto zeros_ite = zeros.begin();

    //             switch(max_num_zero.index()){
    //                 case 0:
    //                 max_num_zero_index = static_cast<int>(std::get<MaxNumZeroRow>(max_num_zero));
    //                 row_lines.push_back(static_cast<int>(max_num_zero_index));
    //                 while(zeros_ite != zeros.end()){
    //                     if((*zeros_ite).y == max_num_zero_index) zeros_ite = zeros.erase(zeros_ite);
    //                     else zeros_ite++;
    //                 }
    //                 break;

    //                 default:
    //                 max_num_zero_index = static_cast<int>(std::get<MaxNumZeroCol>(max_num_zero));
    //                 col_lines.push_back(max_num_zero_index);
    //                 while(zeros_ite != zeros.end()){
    //                     if((*zeros_ite).x == max_num_zero_index)zeros_ite = zeros.erase(zeros_ite);
    //                     else zeros_ite++;
    //                 }
    //                 break;
    //             }
    //         }
    //     }

    //     static cv::Mat calc_subtract_matrix(const cv::Mat matrix){
    //         rcpputils::require_true(matrix.type() == CV_16S);

    //         cv::Mat result = cv::Mat::ones(cv::Size(matrix.cols, matrix.rows), CV_16S);
    //         cv::Mat min_max_loc_mask(cv::Size(matrix.cols, matrix.rows), CV_8U, UINT8_MAX);

    //         std::vector<int> row_lines, col_lines;
    //         get_lines(matrix, row_lines, col_lines);

    //         for(const auto row_line : row_lines){
    //             for(int c = 0; c < matrix.cols; c++){
    //                 result.at<int16_t>(row_line, c)--;
    //                 min_max_loc_mask.at<uint8_t>(row_line, c) = 0U;
    //             }
    //         }

    //         for(const auto col_line : col_lines){
    //             for(int r = 0; r < matrix.rows; r++){
    //                 result.at<int16_t>(r, col_line)--;
    //                 min_max_loc_mask.at<uint8_t>(r, col_line) = 0U;
    //             }
    //         }

    //         double min_val;
    //         cv::minMaxLoc(matrix, &min_val, nullptr, nullptr, nullptr, min_max_loc_mask);

    //         std::cout << "min_val: " << min_val << " mask:\n" << cv::format(min_max_loc_mask, cv::Formatter::FMT_DEFAULT) << std::endl;

    //         rcpputils::require_true(min_val != 0);
            
    //         result *= min_val;

    //         return result;
    //     }

    //     static std::vector<cv::Point> do_assign(cv::Mat matrix){
    //         rcpputils::require_true(matrix.type() == CV_16S);

    //         auto try_assign = [](cv::Mat matrix)->std::optional<std::vector<cv::Point>>{
    //             std::vector<std::optional<int>> zero_rows(matrix.rows, std::nullopt);

    //             for(int c = 0; c < matrix.cols; c++){
    //                 for(const auto& zero : find_zeros(matrix.col(c)))
    //                 if(!zero_rows.at(zero.y).has_value()){
    //                     zero_rows.at(zero.y) = c;
    //                     break;
    //                 }
    //             }

    //             std::vector<cv::Point> result{};
    //             for(int r = 0; r < matrix.rows; r++){
    //                 if(!zero_rows.at(r).has_value()) return {};
    //                 result.push_back(
    //                     cv::Point(*zero_rows.at(r), r)
    //                 );
    //             }

    //             return result;
    //         };

    //         std::optional<std::vector<cv::Point>> assignment = try_assign(matrix);

    //         while(!assignment.has_value()){
    //             matrix -= calc_subtract_matrix(matrix);
    //             assignment = try_assign(matrix);
    //         }

    //         return *assignment;
    //     }

    //     static cv::Mat pre_process(cv::Mat matrix){
    //         rcpputils::require_true(matrix.type() == CV_16S);

    //         for(int r = 0; r < matrix.rows; r++){
    //             double min_val;
    //             cv::minMaxLoc(matrix.row(r), &min_val);
    //             matrix.row(r).forEach<int16_t>([min_val](int16_t& val, const int*){val -= min_val;});
    //         }

    //         for(int c = 0; c < matrix.cols; c++){
    //             double min_val;
    //             cv::minMaxLoc(matrix.col(c), &min_val);
    //             matrix.col(c).forEach<int16_t>([min_val](int16_t& val, const int*){val -= min_val;});
    //         }

    //         return matrix;
    //     }

    //     public:
    //     hungarian() = delete;
    //     hungarian(const hungarian&) = delete;

    //     static std::vector<cv::Point> assign(cv::InputArray input, bool transposition = false){
    //         rcpputils::require_true(input.isMat());
    //         rcpputils::require_true(input.type() == CV_16S);
    //         rcpputils::require_true(input.rows() <= input.cols());

    //         std::vector<cv::Point> result{};
    //         cv::Mat input_matrix = input.getMat();
    //         input_matrix.resize(input_matrix.cols);

    //         pre_process(input_matrix);

    //         auto assignment = do_assign(input_matrix);
    //         for(auto& pair : assignment){
    //             if(pair.y >= input.rows()) pair.y = -1;
    //             if(transposition){
    //                 const double x_buf = pair.x;
    //                 pair.x = pair.y;
    //                 pair.y = x_buf;
    //             }
    //         }
    //         return assignment;
    //     }
    // };

    /*https://kopricky.github.io/code/NetworkFlow/hungarian.html*/
    /*を参照*/
    template<typename T> class Hungarian
    {
    private:
        const int U, V;
        std::vector<vector<int> > graph;
        std::vector<T> dual;
        std::vector<int> alloc, rev_alloc, prev;
        const std::vector<std::vector<T> >& cost;
        int matching_size;
        T diff(const int i, const int j){
            return cost[i][j] - dual[i] - dual[U + j];
        }
        void init_feasible_dual(){
            for(int i = 0; i < U; ++i){
                dual[i] = 0;
                for(int j = 0; j < V; ++j){
                    dual[U + j] = min(dual[U + j], cost[i][j]);
                }
            }
        }
        void construct_graph(){
            for(int i = 0; i < U; ++i){
                for(int j = 0; j < V; ++j){
                    graph[i][j] = (diff(i, j) == 0 && rev_alloc[j] != i);
                }
            }
        }
        bool find_augmenting_path(const int cur, const int prv, int& pos){
            prev[cur] = prv;
            if(cur >= U){
                if(rev_alloc[cur - U] < 0) return true;
                if(find_augmenting_path(rev_alloc[cur - U], cur, pos)){
                    graph[rev_alloc[cur - U]][cur - U] = 1;
                    return true;
                }
            }else{
                const int MX = (alloc[cur] < 0 && pos == U) ? U : V;
                for(int i = 0; i < MX; ++i){
                    if(graph[cur][i] && prev[U + i] < 0 && find_augmenting_path(U + i, cur, pos)){
                        graph[cur][i] = 0, alloc[cur] = i, rev_alloc[i] = cur;
                        return true;
                    }
                }
                if(alloc[cur] < 0 && pos < U){
                    graph[cur][pos] = 0, alloc[cur] = pos, rev_alloc[pos] = cur, prev[U + pos] = cur;
                    return ++pos, true;
                }
            }
            return false;
        }
        void update_dual(const T delta){
            for(int i = 0; i < U; ++i) if(prev[i] >= 0) dual[i] += delta;
            for(int i = U; i < U + V; ++i) if(prev[i] >= 0) dual[i] -= delta;
        }
        void maximum_matching(bool initial=false){
            int pos = initial ? V : U;
            for(bool update = false;; update = false){
                fill(prev.begin(), prev.end(), -1);
                for(int i = 0; i < U; ++i){
                    if(alloc[i] < 0 && find_augmenting_path(i, 2 * U, pos)){
                        update = true, ++matching_size;
                        break;
                    }
                }
                if(!update) break;
            }
        }
        int dfs(const int cur, const int prv, std::vector<int>& new_ver){
            prev[cur] = prv;
            if(cur >= U){
                if(rev_alloc[cur - U] < 0) return cur;
                else return dfs(rev_alloc[cur - U], cur, new_ver);
            }else{
                new_ver.push_back(cur);
                for(int i = 0; i < V; ++i){
                    if(graph[cur][i] && prev[U + i] < 0){
                        const int res = dfs(U + i, cur, new_ver);
                        if(res >= U) return res;
                    }
                }
            }
            return -1;
        }
        int increase_matching(const std::vector<std::pair<int, int> >& vec, std::vector<int>& new_ver){
            for(const auto& e : vec){
                if(prev[e.first] < 0){
                    const int res = dfs(e.first, e.second, new_ver);
                    if(res >= U) return res;
                }
            }
            return -1;
        }
        void hint_increment(int cur){
            while(prev[cur] != 2 * U){
                if(cur >= U){
                    graph[prev[cur]][cur - U] = 0, alloc[prev[cur]] = cur - U, rev_alloc[cur - U] = prev[cur];
                }else{
                    graph[cur][prev[cur] - U] = 1;
                }
                cur = prev[cur];
            }
        }
    public:
        Hungarian(const std::vector<std::vector<T> >& _cost)
             : U((int)_cost.size()), V((int)_cost[0].size()), graph(U, std::vector<int>(U, 1)), dual(U + V, numeric_limits<T>::max()),
                alloc(U, -1), rev_alloc(U, -1), prev(2 * U), cost{_cost}, matching_size(0){
            assert(U >= V);
        }
        std::pair<T, std::vector<int> > solve(){
            init_feasible_dual(), construct_graph();
            bool end = false;
            maximum_matching(true);
            while(matching_size < U){
                std::vector<std::pair<T, int> > cand(V, {numeric_limits<T>::max(), numeric_limits<int>::max()});
                for(int i = 0; i < U; ++i){
                    if(prev[i] < 0) continue;
                    for(int j = 0; j < V; ++j){
                        if(prev[U + j] >= 0) continue;
                        cand[j] = min(cand[j], {diff(i, j), i});
                    }
                }
                while(true){
                    T delta = numeric_limits<T>::max();
                    for(int i = 0; i < V; ++i){
                        if(prev[U + i] >= 0) continue;
                        delta = min(delta, cand[i].first);
                    }
                    update_dual(delta);
                    std::vector<pair<int, int> > vec;
                    std::vector<int> new_ver;
                    for(int i = 0; i < V; ++i){
                        if(prev[U + i] >= 0) continue;
                        if((cand[i].first -= delta) == 0) vec.emplace_back(U + i, cand[i].second);
                    }
                    int res = increase_matching(vec, new_ver);
                    if(res >= U){
                        hint_increment(res);
                        if(++matching_size == U) end = true;
                        else construct_graph();
                        break;
                    }else{
                        for(const int v : new_ver){
                            for(int i = 0; i < V; ++i){
                                if(prev[U + i] >= 0) continue;
                                cand[i] = min(cand[i], {diff(v, i), v});
                            }
                        }
                    }
                }
                if(!end) maximum_matching();
            }
            T total_cost = 0;
            for(int i = 0; i < U; ++i){
                if(alloc[i] < V) total_cost += cost[i][alloc[i]];
                else alloc[i] = -1;
            }
            return make_pair(total_cost, alloc);
        }
    };
}//hungarian