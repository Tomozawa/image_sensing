#pragma once

#include <opencv2/core/mat.hpp>

typedef cv::Matx<double, 3, 3> CameraMatrix;
typedef cv::Vec<double, 5> Distorsion;
typedef cv::Matx33d RMatrix;
typedef cv::Vec3d TVec;