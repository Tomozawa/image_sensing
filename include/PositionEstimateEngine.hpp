#pragma once

#include <camera_matrix.hpp>
#include <utils.hpp>
#include <cmath>

namespace position_estimate_engine{
    enum class EngineType{
        MONO_CAM,
        STEREO_CAM
    };

    struct ScalingFactor{
        double area;
        double distance;
    };

    class PositionEstimateEngineBase{
        private:
        const CameraMatrix camera_matrix;

        double full_scale_atan(double x, double y) const {

            const bool need_to_invert = x < 0;
            return std::atan(y / x) + ((need_to_invert)? std::numbers::pi : 0);
        }

        protected:
        virtual double distance(const double area) const = 0;
        cv::Vec3d direction(const cv::Point2d image_point) const{
            cv::Point2d perspective_point;
            sensing_utils::get_perspective_point(image_point, perspective_point, camera_matrix, {1, 1});

            const double projection_height = std::sqrt(
                std::pow(perspective_point.x, 2)
                + std::pow(perspective_point.y, 2)
            );

            const double theta = std::atan(projection_height);
            double phi = full_scale_atan(perspective_point.x, perspective_point.y);
            if(std::isnan(phi)) phi = 0;

            const cv::Vec3d phi_vec(
                -std::sin(phi),
                std::cos(phi),
                0
            );

            const cv::Vec3d rho_vec(
                std::cos(phi),
                std::sin(phi),
                0
            );

            const cv::Vec3d theta_vec = std::cos(theta) * rho_vec - std::sin(theta) * cv::Vec3d{0, 0, 1};

            const cv::Vec3d r_vec = theta_vec.cross(phi_vec);

            return r_vec;
        }

        public:
        explicit PositionEstimateEngineBase(const CameraMatrix camera_matrix): camera_matrix(camera_matrix){}

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
        private:
        const ScalingFactor scaling_factor;

        public:
        PositionEstimateEngine(const ScalingFactor scaling_factor, const CameraMatrix camera_matrix):
            PositionEstimateEngineBase(camera_matrix),
            scaling_factor(scaling_factor)
        {}

        double distance(const double area) const override{
            return scaling_factor.distance * std::sqrt(scaling_factor.area / area);
        }
    };
}