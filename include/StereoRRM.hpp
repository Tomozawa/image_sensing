#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

namespace stereo_rrm{
    class StereoRRM{
        public:

        static std::vector<double> calc_disps(const std::vector<cv::Point2i>& left_image_points, const std::vector<cv::Point2i>& right_image_points){
            CV_DbgAssert(left_image_points.size() == right_image_points.size());
            std::vector<unsigned> right_permulaion(left_image_points.size());
            std::iota(right_permulaion.begin(), right_permulaion.end(), 1);

            std::vector<double> result;
            double largest_angle_power_sum = 0;
            do{
                double product_power_sum = 0;
                std::vector<double> disps;

                for(unsigned i = 0; i < left_image_points.size(); ++i){
                    const unsigned left_index = i;
                    const unsigned right_index = right_permulaion.at(i);
                    const cv::Vec2d trans_vec(
                        right_image_points.at(right_index).x - left_image_points.at(left_index).x,
                        right_image_points.at(right_index).y - left_image_points.at(left_index).y
                    );
                    const cv::Vec2d epipola_vec(
                        1,
                        0
                    );
                    const double product = trans_vec.dot(epipola_vec);
                    product_power_sum += product * product;
                    disps.at(i) = product / cv::norm(epipola_vec, cv::NORM_L2);
                }

                if(product_power_sum > largest_angle_power_sum){
                    largest_angle_power_sum = product_power_sum;
                    std::copy(disps.begin(), disps.end(), result.begin());
                }
            }while(std::next_permutation(right_permulaion.begin(), right_permulaion.end()));

            return result;
        }
    };
}// stereo_rrm