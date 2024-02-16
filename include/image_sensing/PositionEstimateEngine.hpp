#pragma once

#include <camera_matrix.hpp>
#include <utils.hpp>
#include <cmath>

namespace position_estimate_engine{
    enum class EngineType{
        MONO_CAM,
        STEREO_CAM
    };

    struct PointingFactor{
        double focal_distance;
        cv::Size image_size;
    };

    class PositionEstimateEngineBase{
        protected:
        const PointingFactor pointing_factor;
        const CameraMatrix camera_matrix;

        virtual double distance(const double area) const = 0;
        cv::Vec3d direction(const cv::Point2d image_point) const{
            cv::Point2d perspective_point;
            sensing_utils::get_perspective_point(image_point, perspective_point, camera_matrix);

            const double phi = sensing_utils::fullscale_atan(perspective_point.x, perspective_point.y);
            const double theta = std::atan(cv::norm(perspective_point));

            const cv::Vec3d phi_vec(
                -std::sin(phi),
                std::cos(phi)
            );

            const cv::Vec3d rho_vec(
                std::cos(phi),
                std::sin(phi)
            );

            const cv::Vec3d theta_vec = std::cos(theta) * rho_vec - std::sin(theta) * cv::Vec3d{0, 0, 1};

            const cv::Vec3d r_vec = theta_vec.cross(phi_vec);

            return r_vec;
        }

        public:
        explicit PositionEstimateEngineBase(const PointingFactor pointing_factor, const CameraMatrix camera_matrix): pointing_factor(pointing_factor), camera_matrix(camera_matrix){}

        cv::Point3d estimate_position(const cv::Point2d image_point, const float area) const{
            const double distance = this->distance(area);
            const cv::Vec3d direction = this->direction(image_point);
            return cv::Point3d{distance * direction[0], distance * direction[1], distance * direction[2]};
        }

    };

    template<EngineType type>
    class PositionEstimateEngine{};

    template<>
    class PositionEstimateEngine<EngineType::MONO_CAM> : public PositionEstimateEngineBase{
        public:
        PositionEstimateEngine(const PointingFactor pointing_factor, const CameraMatrix camera_matrix):
            PositionEstimateEngineBase(pointing_factor, camera_matrix)
        {}

        double distance(const double area) const override{
            constexpr double ball_diameter_mm = 190;
            const double ball_horizontal_radius = (ball_diameter_mm / 2.0 /pointing_factor.focal_distance) * camera_matrix(0, 0);
            const double ball_vertical_radius = (ball_diameter_mm / 2.0 / pointing_factor.focal_distance) * camera_matrix(1, 1);
            const double standard_ball_area = std::numbers::pi * ball_horizontal_radius * ball_vertical_radius;

            const double normalized_distance = std::sqrt(standard_ball_area / area);

            return normalized_distance * pointing_factor.focal_distance;
        }
    };
}