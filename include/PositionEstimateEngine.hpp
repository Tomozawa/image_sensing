#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <matrix.hpp>
#include <utils.hpp>
#include <cmath>
#include <StereoRRM.hpp>

namespace position_estimate_engine{
    enum class EngineType{
        MONO_CAM,
        STEREO_CAM_RRM, // Round Robin Matching
        STEREO_CAM_SGBM
    };

    struct ScalingFactor{
        double area;
        double distance;
    };

    template<EngineType type>
    class PositionEstimateEngine{
    };

    template<>
    class PositionEstimateEngine<EngineType::MONO_CAM>{
        private:
        const ScalingFactor scaling_factor;
        const CameraMatrix camera_matrix;

        public:
        PositionEstimateEngine(const ScalingFactor scaling_factor, const CameraMatrix camera_matrix):
            scaling_factor(scaling_factor)
        {}

        cv::Vec3d direction(const cv::Point2d image_point) const{
            cv::Point2d perspective_point;
            sensing_utils::get_perspective_point(image_point, perspective_point, camera_matrix, {1, 1});

            const double projection_height = std::sqrt(
                std::pow(perspective_point.x, 2)
                + std::pow(perspective_point.y, 2)
            );

            const double theta = std::atan(projection_height);
            double phi = sensing_utils::full_scale_atan(perspective_point.x, perspective_point.y);
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

        double distance(const double area) const{
            return scaling_factor.distance * std::sqrt(scaling_factor.area / area);
        }

        /*ひずみ補正済みの画像を入力する*/
        cv::Point3d estimate_position(const cv::Point2i image_point, const float area) const{
            const double distance = this->distance(area);
            const cv::Vec3d direction = this->direction(image_point);
            return cv::Point3d{distance * direction[0], distance * direction[1], distance * direction[2]};
        }
    };

    template<>
    class PositionEstimateEngine<EngineType::STEREO_CAM_SGBM>{
        private:
        const cv::Ptr<cv::StereoSGBM> sgbm;
        const cv::Mat mapl1;
        const cv::Mat mapl2;
        const cv::Mat mapr1;
        const cv::Mat mapr2;
        const cv::Matx44d Q;

        public:
        explicit PositionEstimateEngine(
            const cv::Ptr<cv::StereoSGBM> sgbm,
            const cv::Mat mapl1,
            const cv::Mat mapl2,
            const cv::Mat mapr1,
            const cv::Mat mapr2,
            const QMatrix Q
        ):
            sgbm(sgbm),
            mapl1(mapl1),
            mapl2(mapl2),
            mapr1(mapr1),
            mapr2(mapr2),
            Q(Q)
        {}

        static PositionEstimateEngine<EngineType::STEREO_CAM_SGBM> create(
            const cv::Size2i img_size,
            const unsigned num_channels,
            const CameraMatrix lcamera_matrix,
            const CameraMatrix rcamera_matrix,
            const RMatrix R,
            const TVec T
        ){
            /*const unsigned numDisparties = static_cast<unsigned>((img_size.width / 8) + 15) & ~0b1111*/;
            const unsigned numDisparties = 150;
            const unsigned SADWindowSize = 6;
            const auto sgbm = cv::StereoSGBM::create(
                0,
                numDisparties,
                SADWindowSize,
                /*8U * num_channels * SADWindowSize * SADWindowSize*/8U * 3 * num_channels * SADWindowSize * SADWindowSize,
                /*32U * num_channels * SADWindowSize * SADWindowSize*/32U * 3 * num_channels * SADWindowSize * SADWindowSize,
                /*1*/11,
                /*63*/0,
                5,
                100,
                /*32*/16
            );

            cv::Mat Rl, Rr, Pl, Pr, mapl1, mapl2, mapr1, mapr2;
            QMatrix Q;
            cv::stereoRectify(
                lcamera_matrix,
                cv::Vec4d(),
                rcamera_matrix,
                cv::Vec4d(),
                img_size,
                R,
                T,
                Rl,
                Rr,
                Pl,
                Pr,
                Q,
                0   
            );
            cv::initUndistortRectifyMap(
                lcamera_matrix,
                cv::Vec4d(),
                Rl,
                Pl,
                img_size,
                CV_16SC2,
                mapl1,
                mapl2
            );
            cv::initUndistortRectifyMap(
                rcamera_matrix,
                cv::Vec4d(),
                Rr,
                Pr,
                img_size,
                CV_16SC2,
                mapr1,
                mapr2
            );

            return PositionEstimateEngine<EngineType::STEREO_CAM_SGBM>(
                std::move(sgbm),
                mapl1,
                mapl2,
                mapr1,
                mapr2,
                Q
            );
        }

        /*ひずみ補正済み・平行化前の画像を入力する*/
        cv::Mat get_disp(const cv::InputArray left, const cv::InputArray right) const{
            cv::Mat serialized_left, serialized_right;
            cv::remap(
                left,
                serialized_left,
                mapl1,
                mapl2,
                cv::INTER_LINEAR
            );
            cv::remap(
                right,
                serialized_right,
                mapr1,
                mapr2,
                cv::INTER_LINEAR
            );
            
            cv::Mat disp;
            sgbm->compute(
                serialized_left,
                serialized_right,
                disp
            );
            return disp;
        }

        cv::Point3d estimate_position(const cv::Mat disp, const cv::Point2i image_point) const{
            CV_DbgAssert(disp.type() == CV_16S);
            const cv::Vec4d homo_2d_vec(
                image_point.x,
                image_point.y,
                static_cast<double>(disp.at<short>(image_point.y, image_point.x)) / 16.0,
                1
            );
            const cv::Vec4d homo_3d_vec = Q * homo_2d_vec;

            return cv::Point3d(
                homo_3d_vec[0] / homo_3d_vec[3],
                homo_3d_vec[1] / homo_3d_vec[3],
                homo_3d_vec[2] / homo_3d_vec[3]
            );
        }
    };

    template<>
    class PositionEstimateEngine<EngineType::STEREO_CAM_RRM>{
        private:
        const QMatrix Q;

        public:
        explicit PositionEstimateEngine(const QMatrix& Q): Q(Q){}

        /*ひずみ補正済み・平衡化済みの画像を入力する*/
        std::vector<cv::Point3d> estimate_positions(const std::vector<cv::Point2i>& left_image_points, const std::vector<cv::Point2i>& right_image_points) const{
            std::vector<cv::Point3d> result;
            std::vector<double> disps = stereo_rrm::StereoRRM::calc_disps(left_image_points, right_image_points);
            for(unsigned i = 0; i < left_image_points.size(); ++i){
                const cv::Vec4d homo_2d_vec(
                left_image_points.at(i).x,
                left_image_points.at(i).y,
                disps.at(i),
                1
            );
            const cv::Vec4d homo_3d_vec = Q * homo_2d_vec;

            result.push_back(
                {
                    homo_3d_vec[0] / homo_3d_vec[3],
                    homo_3d_vec[1] / homo_3d_vec[3],
                    homo_3d_vec[2] / homo_3d_vec[3]
                }
            );
            }
            return result;
        }
    };
}