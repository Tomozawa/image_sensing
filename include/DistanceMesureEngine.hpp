#pragma once

#include <camera_matrix.hpp>
#include <cmath>

namespace distance_mesure_engine{
    struct ScalingFactor{
        float area;
        float distance;
    };

    enum class EngineType{
        MONO_CAMERA,
        STEREO_CAMERA
    };

    template<EngineType type>
    class DistanceMesureEngine{};

    template<>
    class DistanceMesureEngine<EngineType::MONO_CAMERA>{
    private:
        const CameraMatrix inverted_internal_camera_matrix;
        const ScalingFactor scaling_factor;
        const cv::Vec2d optical_axis_offset;

        double full_scale_atan(const double x, const double y) const {
            const bool need_to_invert = x < 0;

            return std::atan(y / x) + (need_to_invert)? std::numbers::pi : 0;
        }

    public:
        DistanceMesureEngine(const CameraMatrix internal_camera_matrix, const ScalingFactor scaling_factor):
            inverted_internal_camera_matrix(internal_camera_matrix.inv()),
            scaling_factor(scaling_factor),
            optical_axis_offset(cv::Vec2d(internal_camera_matrix(0, 2), internal_camera_matrix(1, 2)))
        {}

        cv::Point3f estimate_depth(const cv::Point2d image_point, const float area) const {
            const cv::Vec3d image_point_vec(image_point.x, image_point.y, 1);
            const cv::Matx camera_point = inverted_internal_camera_matrix * image_point_vec;
            const double obj_distance = scaling_factor.distance * (area / scaling_factor.area);
            const double project_point_distance = std::pow(camera_point(0), 2) + std::pow(camera_point(1), 2);
        }
    };
}