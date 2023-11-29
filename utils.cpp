#include <utils.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;

namespace sensing_utils{
    void hsv_range(InputArray input, const Vec<uint8_t, 256>& hue_lut, int s_min, int s_max, int v_min, int v_max, OutputArray output){
        CV_DbgAssert(input.isMat());

        Mat split_hsv[3];
        Mat binaried_hsv[3];
        split(input.getMat(), split_hsv);
        LUT(split_hsv[0], hue_lut, binaried_hsv[0]);
        inRange(split_hsv[1], cv::Scalar{static_cast<double>(s_min)}, cv::Scalar{static_cast<double>(s_max)}, binaried_hsv[1]);
        inRange(split_hsv[2], cv::Scalar{static_cast<double>(v_min)}, cv::Scalar{static_cast<double>(v_max)}, binaried_hsv[2]);

        Mat sv;
        bitwise_and(binaried_hsv[1], binaried_hsv[2], output, binaried_hsv[0]);
    }

    void get_perspective_point(const cv::Point2d image_point, cv::Point2d& perspective_point, const CameraMatrix camera_matrix, const Scalar magnification){
        const cv::Vec2d image_point_vec(image_point.x, image_point.y);
        const cv::Vec2d opt_center_vec(camera_matrix(0, 2), camera_matrix(1, 2));
        const cv::Scalar focal_px(camera_matrix(0, 0), camera_matrix(1, 1));
        cv::Vec2d perspective_point_vec = image_point_vec - opt_center_vec;

        perspective_point_vec[0] /= focal_px[0];
        perspective_point_vec[1] /= focal_px[1];

        perspective_point.x = perspective_point_vec[0] * magnification[0];
        perspective_point.y = perspective_point_vec[1] * magnification[1];
    }
} //utils