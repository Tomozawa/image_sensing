#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <optional>

namespace stereo_rrm{
    class StereoRRM{
        public:

        static std::optional<std::vector<double>> calc_disps(const std::vector<cv::Point2i>& left_image_points, const std::vector<cv::Point2i>& right_image_points){
            CV_DbgAssert(left_image_points.size() == right_image_points.size());
            if(left_image_points.size() > 10) return {};

            std::vector<unsigned> right_permulaion(left_image_points.size());
            std::iota(right_permulaion.begin(), right_permulaion.end(), 0);

            std::vector<double> result(left_image_points.size());
            double largest_angle_power_sum = 0;
            do{
                double angle_power_sum = 0;
                std::vector<double> disps;

                for(unsigned i = 0; i < left_image_points.size(); ++i){
                    const unsigned left_index = i;
                    const unsigned right_index = right_permulaion.at(i);
                    const cv::Vec2d left_image_point_vec(
                        left_image_points.at(i).x,
                        left_image_points.at(i).y
                    );
                    const cv::Vec2d right_image_point_vec(
                        right_image_points.at(i).x,
                        right_image_points.at(i).y
                    );
                    const cv::Vec2d trans_vec = right_image_point_vec - left_image_point_vec;
                    const cv::Vec2d epipola_vec(
                        1,
                        0
                    );
                    const double product = trans_vec.dot(epipola_vec);
                    const double normalized_product = product / cv::norm(trans_vec, cv::NORM_L2) / cv::norm(epipola_vec, cv::NORM_L2);
                    angle_power_sum += normalized_product * normalized_product;
                    disps.push_back(cv::norm(trans_vec, cv::NORM_L2));
                }

                if(angle_power_sum > largest_angle_power_sum){
                    largest_angle_power_sum = angle_power_sum;
                    std::copy(disps.begin(), disps.end(), result.begin());
                }
            }while(std::next_permutation(right_permulaion.begin(), right_permulaion.end()));

            return result;
        }
    };
}// stereo_rrm