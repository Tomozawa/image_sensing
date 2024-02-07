#include <opencv2/core/utils/logger.hpp>
#include <StereoRRM.hpp>
#include <iostream>

using namespace cv;
using namespace stereo_rrm;

int main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);

#ifdef _DEBUG
    CV_LOG_DEBUG(nullptr, "In Debug mode");
#endif

    std::vector<Point2i> left_image_points = {
        {1, 0},
        {2, 1},
        {3, 2},
        {4, 3}
    };
    std::vector<Point2i> right_image_points;
    for(auto point : left_image_points){
        right_image_points.push_back({
            point.x + 1,
            point.y
        });
    }
    // const auto disps = StereoRRM::calc_disps(left_image_points, right_image_points);

    // std::cout << "(";
    // for(auto disp : disps){
    //     std::cout << disp << ", ";
    // }
    // std::cout << "\b\b)";
}