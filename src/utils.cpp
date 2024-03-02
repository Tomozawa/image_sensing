#include <utils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <cmath>

using namespace cv;

namespace sensing_utils{
    void get_perspective_point(const cv::Point2d image_point, cv::Point2d& perspective_point, const CameraMatrix camera_matrix){
        const cv::Vec3d image_point_vec(image_point.x, image_point.y, 1);
        const cv::Vec3d perspective_point_vec = camera_matrix.inv() * image_point_vec;

        perspective_point.x = perspective_point_vec[0];
        perspective_point.y = perspective_point_vec[1];
    }

    double fullscale_atan(const double x, const double y){
        return std::fmod(((x >= 0)? std::atan(y / x) : (std::atan(y / x) + std::numbers::pi)) + 2.0 * std::numbers::pi, 2.0 * std::numbers::pi);
    }

    bool equals(const Mat& mat1, const Mat& mat2){
        rcpputils::require_true(mat1.type() == CV_16U && mat2.type() == CV_16U);

        if(mat1.rows != mat2.rows || mat1.cols != mat2.cols) return false;

        for(int r = 0; r < mat1.rows; r++){
            for(int c = 0; c < mat1.cols; c++){
                if(mat1.at<uint16_t>(r, c) != mat2.at<uint16_t>(r, c)) return false;
            }
        }

        return true;
    }
} //sensing_utils