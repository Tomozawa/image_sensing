#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
#include <fstream>
#include <exception>
#include <string>
#include <chrono>
#include <functional>
#include <PositionEstimateEngine.hpp>
#include <InRangeParams.hpp>
#include <HSVLookupTable.hpp>
#include <utils.hpp>
#include <IdentifyEngine.hpp>
#include <nlohmann/json.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rcpputils/asserts.hpp>
#include <nhk24_utils/msg/balls.hpp>
#include <geometry_msgs/msg/point.hpp>

using namespace cv;
using namespace in_range_params;
using namespace sensing_utils;
using namespace hue_lut;
using namespace position_estimate_engine;
using namespace identify_engine;
using namespace nlohmann;
using namespace std::chrono_literals;

using LUT_TYPE = LUT_TYPE_;
using LUT_PARAM = LUT_PARAM_<3>;
using LUT_VALUE_PAIR = LUT_VALUE_PAIR_;

struct CameraCalibration{
    CameraMatrix camera_matrix;
    Vec<double, 5> distorsion;
    double focal_distance;
    Size image_size;
};

enum class ColorID : nhk24_utils::msg::Ball::_color_type{
    PURPLE,
    RED,
    BLUE
};

inline void prepare_image(InputArray, OutputArray, const CameraMatrix&, const Vec<double, 5>&);
std::vector<std::pair<nhk24_utils::msg::Ball::_id_type, nhk24_utils::msg::Ball::_position_type>> find_balls(InputArray, const LUT_TYPE&, const PositionEstimateEngine<EngineType::MONO_CAM>&, IdentifyEngine&);
inline CameraCalibration load_calibration_file(void);
inline VideoCapture open_cameras(void);

template<InRangeParams paddy_params, InRangeParams empty_params, ColorID color_id>
class Application final : public rclcpp::Node{
    private:
    VideoCapture video_capture;
    const CameraCalibration calibration;
    const PositionEstimateEngine<EngineType::MONO_CAM> position_estimate_engine;
    IdentifyEngine paddy_identify_engine;
    IdentifyEngine empty_identify_engine;
    std::shared_ptr<rclcpp::Publisher<nhk24_utils::msg::Balls>> publisher;
    rclcpp::TimerBase::SharedPtr timer;
    const LUT_TYPE paddy_lut;
    const LUT_TYPE empty_lut;

    public:
    Application():
        rclcpp::Node("ball_detecter"),
        video_capture(open_cameras()),
        calibration(load_calibration_file()),
        position_estimate_engine(
            PointingFactor{
                .focal_distance = calibration.focal_distance,
                .image_size = calibration.image_size
            },
            calibration.camera_matrix
        ),
        paddy_identify_engine(),
        empty_identify_engine(),
        publisher(create_publisher<nhk24_utils::msg::Balls>("balls", 256)),
        timer(create_wall_timer(16ms, std::bind(&Application::loop, this))),
        paddy_lut(calc_lut(
            LUT_PARAM{
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(paddy_params.h_min), static_cast<uint8_t>(paddy_params.h_max)
                },
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(paddy_params.s_min), static_cast<uint8_t>(paddy_params.s_max)
                },
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(paddy_params.v_min), static_cast<uint8_t>(paddy_params.v_max)
                }
            }
        )),
        empty_lut(calc_lut(
            LUT_PARAM{
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(empty_params.h_min), static_cast<uint8_t>(empty_params.h_max)
                },
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(empty_params.s_min), static_cast<uint8_t>(empty_params.s_max)
                },
                LUT_VALUE_PAIR{
                    static_cast<uint8_t>(empty_params.v_min), static_cast<uint8_t>(empty_params.v_max)
                }
            }
        ))
    {
        utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_WARNING);
        
        const bool is_camera_opened = video_capture.isOpened() && video_capture.get(CAP_PROP_FRAME_WIDTH) == calibration.image_size.width && video_capture.get(CAP_PROP_FRAME_HEIGHT) == calibration.image_size.height;
        if(!is_camera_opened) rclcpp::shutdown();
    }

    void loop();
};

template<InRangeParams paddy_params, InRangeParams empty_params, ColorID color_id>
void Application<paddy_params, empty_params, color_id>::loop(){
    Mat input_img, prepared_img;
    std::vector<std::vector<Point>> contours{};

    if(!video_capture.read(input_img)) return;

    prepare_image(input_img, prepared_img, calibration.camera_matrix, calibration.distorsion);

    const auto paddy_balls = find_balls(prepared_img, paddy_lut, position_estimate_engine, paddy_identify_engine);
    const auto empty_balls = find_balls(prepared_img, empty_lut, position_estimate_engine, empty_identify_engine);

    nhk24_utils::msg::Balls message{};
    for(const auto& paddy_ball : paddy_balls){
        message.balls.push_back(
            nhk24_utils::msg::Balls::_balls_type::value_type().set__color(static_cast<std::underlying_type_t<ColorID>>(color_id)).set__id(paddy_ball.first).set__position(paddy_ball.second)
        );
    }
    for(const auto& empty_ball : empty_balls){
        message.balls.push_back(
            nhk24_utils::msg::Balls::_balls_type::value_type().set__color(static_cast<std::underlying_type_t<ColorID>>(ColorID::PURPLE)).set__id(empty_ball.first).set__position(empty_ball.second)
        );
    }
    if(message.balls.size() > 0) publisher->publish(message);

}

void prepare_image(InputArray input, OutputArray output, const CameraMatrix& camera_matrix, const Vec<double, 5>& dist){
    rcpputils::require_true(input.isMat());
    Mat image = input.getMat();
    Mat undist_img, hsv, blur;
    undistort(image, undist_img, camera_matrix, dist);
    cvtColor(undist_img, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(11, 11), 8.5, 8.5);

    output.move(blur);
}

std::vector<std::pair<nhk24_utils::msg::Ball::_id_type, nhk24_utils::msg::Ball::_position_type>> find_balls(InputArray input, const LUT_TYPE& lut, const PositionEstimateEngine<EngineType::MONO_CAM>& position_estimate_engine, IdentifyEngine& identify_engine){
    Mat image, hsv_filtered, closed, opened, canny_img;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    rcpputils::require_true(input.isMat());
    image = input.getMat();

    //普通のinRangeだと赤色が検知できない
    // hsv_range(blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);
    LUT(image, lut, hsv_filtered);

    opening<2>(hsv_filtered, opened);
    closing<2>(opened, closed);
    
    Canny(closed, canny_img, 25, 75);

    findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    std::vector<Point> ball_moment_points{};
    std::vector<Point3d> ball_positions{};
    std::vector<uint16_t> ball_ids;
    for(size_t i = 0; i < contours.size(); i++){
        if(hierarchy.at(i)[3] >= 0) continue;

        std::vector<Point> applox_contour, convex_contour;
        approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
        convexHull(applox_contour, convex_contour);

        const double area = contourArea(convex_contour);

        if(area < 400) continue;
        if(convex_contour.size() < 10) continue;

        const Moments ball_moments = moments(convex_contour);

        ball_moment_points.push_back(
            Point(
                static_cast<int>(ball_moments.m10 / ball_moments.m00),
                static_cast<int>(ball_moments.m01 / ball_moments.m00)
            )
        );
        ball_positions.push_back(
            position_estimate_engine.estimate_position(
                Point2d(
                    ball_moments.m10 / ball_moments.m00,
                    ball_moments.m01 / ball_moments.m00
                ),
                area
            )
        );
    }
    ball_ids = identify_engine.identify(ball_moment_points);

    rcpputils::assert_true(ball_positions.size() == ball_ids.size());
    std::vector<std::pair<nhk24_utils::msg::Ball::_id_type, nhk24_utils::msg::Ball::_position_type>> result{};
    for(size_t i = 0U; i < ball_positions.size(); i++){
        result.push_back(
            std::pair<nhk24_utils::msg::Ball::_id_type, nhk24_utils::msg::Ball::_position_type>(
                ball_ids.at(i),
                nhk24_utils::msg::Ball::_position_type().set__x(ball_positions.at(i).x).set__y(ball_positions.at(i).y).set__z(ball_positions.at(i).z)
            )
        );
    }

    return result;
}

CameraCalibration load_calibration_file(){
    CameraCalibration result;

    std::ifstream ifs("./src/image_sensing/src/camera_calibration.json");
    json calibration_json;
    if(!ifs.is_open()) throw std::runtime_error("Can't open camera_calibration.json");

    ifs >> calibration_json;

    ifs.close();

    const json& matrix_ref = calibration_json.at("matrix");
    result.camera_matrix = {
        matrix_ref.at(0).at(0), matrix_ref.at(0).at(1), matrix_ref.at(0).at(2),
        matrix_ref.at(1).at(0), matrix_ref.at(1).at(1), matrix_ref.at(1).at(2),
        matrix_ref.at(2).at(0), matrix_ref.at(2).at(1), matrix_ref.at(2).at(2)
    };

    const json& distortion_ref = calibration_json.at("distortion");
    result.distorsion = {
        distortion_ref.at(0).at(0),
        distortion_ref.at(0).at(1),
        distortion_ref.at(0).at(2),
        distortion_ref.at(0).at(3),
        distortion_ref.at(0).at(4)
    };

    result.focal_distance = calibration_json.at("focal_distance");

    const json& image_size_ref = calibration_json.at("image_size");
    result.image_size = Size(image_size_ref.at(0), image_size_ref.at(1));

    return result;
}

VideoCapture open_cameras(void){
    VideoCapture result;
    result.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    result.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    result.open(14, cv::CAP_V4L2);
    return result;
}

int main(int argc, char* argv[]){
    using BlueBallApplication = Application<
        InRangeParams{
            .h_min = 128,
            .s_min = 70,
            .v_min = 92,
            .h_max = 185,
            .s_max = 239,
            .v_max = 255
        },
        InRangeParams{
            .h_min = 0,
            .s_min = 0,
            .v_min = 0,
            .h_max = 255,
            .s_max = 255,
            .v_max = 255
        },
        ColorID::BLUE
    >;

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BlueBallApplication>());
    rclcpp::shutdown();
    return 0;
}