#pragma once

#include <tuple>

#include <Hungarian.hpp>

namespace identify_engine{
    class IdentifyEngine final{
        private:
        std::vector<std::tuple<uint16_t, cv::Point>> last_balls;
        uint16_t next_id;

        bool calc_cost_matrix(std::vector<cv::Point>& balls, std::vector<std::vector<uint16_t>>& result){
            if(balls.size() >= last_balls.size()){
                for(int b = 0; b < balls.size(); b++){
                    result.push_back(std::vector<uint16_t>());
                    for(int last_b = 0; last_b < last_balls.size(); last_b++){
                        result.at(result.size() - 1U).push_back(static_cast<uint16_t>(cv::norm(balls.at(b) - std::get<1>(last_balls.at(last_b)))));
                    }
                }
                return false;
            }else{
                for(int last_b = 0; last_b < last_balls.size(); last_b++){
                    result.push_back(std::vector<uint16_t>());
                    for(int b = 0; b < balls.size(); b++){
                        result.at(result.size() - 1U).push_back(static_cast<uint16_t>(cv::norm(balls.at(b) - std::get<1>(last_balls.at(last_b)))));
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
                std::vector<std::vector<uint16_t>> cost_matrix;
                const bool is_transposition = calc_cost_matrix(balls, cost_matrix);

                const auto assignments = hungarian::Hungarian<uint16_t>(cost_matrix).solve().second;

                for(int b = 0; b < assignments.size(); b++){
                    const int last_b = assignments.at(b);

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