#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <camera_matrix.hpp>

#include <geometry_msgs/msg/point.hpp>

#include <rcpputils/asserts.hpp>

namespace sensing_utils{
    template<unsigned char DilateErodeTimes>
    void closing(cv::InputArray input, cv::OutputArray output){
        
        rcpputils::require_true(input.isMat());

        cv::Mat mat_buf[3] = {cv::Mat(input.getMat()), cv::Mat(), cv::Mat()};
        const cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3));

        //closing
        cv::dilate(mat_buf[0], mat_buf[1], kernel, cv::Point(-1, -1), DilateErodeTimes);
        cv::erode(mat_buf[1], mat_buf[2], kernel, cv::Point(-1, -1), DilateErodeTimes);

        output.move(mat_buf[2]);
    }

    template<unsigned char DilateErodeTimes>
    void opening(cv::InputArray input, cv::OutputArray output){
        
        rcpputils::require_true(input.isMat());

        cv::Mat mat_buf[3] = {cv::Mat(input.getMat()), cv::Mat(), cv::Mat()};
        cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3));

        //opening
        cv::erode(mat_buf[0], mat_buf[1], kernel, cv::Point(-1, -1), DilateErodeTimes);
        cv::dilate(mat_buf[1], mat_buf[2], kernel, cv::Point(-1, -1), DilateErodeTimes);

        output.move(mat_buf[2]);
    }

    void hsv_range(cv::InputArray, const cv::Vec<uint8_t, 256>&, int, int, int, int, cv::OutputArray);

    void get_perspective_point(const cv::Point2d, cv::Point2d&, const CameraMatrix);

    double fullscale_atan(const double, const double);
} //sensing_utils