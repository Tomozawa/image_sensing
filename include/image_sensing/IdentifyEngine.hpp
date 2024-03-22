#pragma once

#include <tuple>

#include <Hungarian.hpp>

namespace identify_engine{
    class IdentifyEngine final{
        private:
        std::vector<std::tuple<uint16_t, cv::Point>> last_balls;
        uint16_t next_id;

        bool calc_cost_matrix(std::vector<cv::Point>& balls, std::vector<std::vector<uint8_t>>& result){
            if(balls.size() < last_balls.size()){
                result.create(balls.size(), last_balls.size(), CV_16S);
                for(int b = 0; b < result.rows; b++){
                    for(int last_b = 0; last_b < result.cols; last_b++){
                        result.at<int16_t>(b, last_b) = static_cast<int16_t>(cv::norm(balls.at(b) - std::get<1>(last_balls.at(last_b))));
                    }
                }
                return false;
            }else{
                result.create(last_balls.size(), balls.size(), CV_16S);
                for(int b = 0; b < result.cols; b++){
                    for(int last_b = 0; last_b < result.rows; last_b++){
                        result.at<int16_t>(last_b, b) = static_cast<int16_t>(cv::norm(balls.at(b) - std::get<1>(last_balls.at(last_b))));
                    }
                }
                return true;
            }
        }

        public:
        IdentifyEngine():
            last_balls(),
            next_id(0)
        {}

        std::vector<uint16_t> identify(std::vector<cv::Point>& balls){
            std::vector<uint16_t> result(balls.size());

            if(last_balls.size() > 0){
                cv::Mat cost_matrix;
                const bool is_transposition = calc_cost_matrix(balls, cost_matrix);

                const std::vector<cv::Point> assignments = hungarian::hungarian::assign(cost_matrix, is_transposition);

                for(const auto& assignment : assignments){
                    if(assignment.y < 0) continue;
                    
                    if(assignment.x >= 0){
                        result.at(assignment.y) = std::get<0>(last_balls.at(assignment.x));
                    }
                    else result.at(assignment.y) = next_id++;
                }
            }else{
                for(auto& elem : result){
                    elem = next_id++;
                }
            }

            last_balls.clear();

            for(size_t i = 0U; i < balls.size(); i++){
                last_balls.push_back(std::make_tuple(result.at(i), balls.at(i)));
            }

            return result;
        }
    };
}//idendity_engine