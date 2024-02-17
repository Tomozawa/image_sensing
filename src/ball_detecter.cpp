#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/calib3d.hpp>
#include <memory>
#include <vector>
#include <fstream>
#include <exception>
#include <string>
#include <chrono>
#include <functional>
#include <PositionEstimateEngine.hpp>
#include <InRangeParams.hpp>
#include <HueLookupTable.hpp>
#include <utils.hpp>
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
using namespace nlohmann;
using namespace std::chrono_literals;


class Application final : public rclcpp::Node{
    private:
    struct CameraCalibration{
        CameraMatrix camera_matrix;
        Vec<double, 5> distorsion;
        double focal_distance;
        Size image_size;
    };

    struct GlobalVariables{
        private:
            InRangeParams params_ = {0, 0, 0, 0, 0, 0};
            VideoCapture video_capture_;
            LUT_TYPE hue_lut_;
        public:
            /* params_ */
            //Todo: 値チェック
            //Todo: インデックスをテンプレート化または列挙子にする
            //取得と設定を同時に行うことを強制する(取得のみすることもできる)
            //取得(ゲット)→更新→更新の適用(セット)を呼び出し元に任せると、更新の適用忘れがあるかもしれないので
            inline InRangeParams replace_param(const InRangeParams::ValueType val, int8_t index){
                switch(index){
                    case 0: 
                        params_.h_min = val;
                        calc_hue_lut(params_.h_min, params_.h_max, hue_lut_);
                        break;
                    case 1:
                        params_.h_max = val;
                        calc_hue_lut(params_.h_min, params_.h_max, hue_lut_);
                        break;
                    case 2: params_.s_min = val; break;
                    case 3: params_.s_max = val; break;
                    case 4: params_.v_min = val; break;
                    case 5: params_.v_max = val; break;
                }
                return params_;
            }
            inline InRangeParams replace_param(){return params_;}

            /*video_capture_*/
            //Todo: 複数カメラ対応
            inline bool open_cameras(void){
                video_capture_.open(1);
                return video_capture_.isOpened();
            }
            //Todo: 複数カメラ対応(1つのメソッドで全部grab)
            inline bool grabs(void){return video_capture_.grab();}
            //Todo: 複数カメラ対応(テンプレートでカメラ毎)
            inline bool retrieve(OutputArray output, int flag=0){return video_capture_.retrieve(output, flag);}
            //デバッグ用
            inline bool read(OutputArray output){return video_capture_.read(output);}

            /*lut*/
            inline LUT_TYPE get_hue_lut(void){return hue_lut_;}
    };

    GlobalVariables global_variables;
    const CameraCalibration calibration;
    const PositionEstimateEngine<EngineType::MONO_CAM> engine;
    std::shared_ptr<rclcpp::Publisher<nhk24_utils::msg::Balls>> publisher;

    void execute_calc(InputArray, nhk24_utils::msg::Balls&, const InRangeParams&, const PositionEstimateEngine<EngineType::MONO_CAM>&);
    CameraCalibration load_calibration_file(void);

    public:
    Application():
        rclcpp::Node("image_sensing"),
        global_variables(),
        calibration(load_calibration_file()),
        engine(
            PointingFactor{
                .focal_distance = calibration.focal_distance,
                .image_size = calibration.image_size
            },
            calibration.camera_matrix
        ),
        publisher(create_publisher<nhk24_utils::msg::Balls>("balls", 256))
    {
        utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);
        
        rcpputils::assert_true(global_variables.open_cameras());

        create_wall_timer(16ms, std::bind(&Application::loop, this));
    }

    void loop();
};

void Application::loop(){
    global_variables.grabs();
    Mat input_img, undistort_img;
    nhk24_utils::msg::Balls message;

    global_variables.retrieve(input_img);

    undistort(input_img, undistort_img, calibration.camera_matrix, calibration.distorsion);

    execute_calc(undistort_img, message, global_variables.replace_param(), engine);

    publisher->publish(message);
}

void Application::execute_calc(InputArray input, nhk24_utils::msg::Balls& message, const InRangeParams& params, const PositionEstimateEngine<EngineType::MONO_CAM>& engine){
    Mat image, hsv, blur, hsv_filtered, closed, opened, canny_img;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    rcpputils::require_true(input.isMat());
    image = input.getMat();

    cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(11, 11), 8.5, 8.5);

    //普通のinRangeだと赤色が検知できない
    hsv_range(blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

    opening<2>(hsv_filtered, opened);
    closing<2>(opened, closed);
    
    Canny(closed, canny_img, 25, 75);

    findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    message = nhk24_utils::msg::Balls();
    for(size_t i = 0; i < contours.size(); i++){
        nhk24_utils::msg::Ball ball;
        if(hierarchy.at(i)[3] >= 0) continue;

        std::vector<Point> applox_contour, convex_contour;
        approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
        convexHull(applox_contour, convex_contour);

        const double area = contourArea(convex_contour);

        if(area < 400) continue;
        if(convex_contour.size() < 10) continue;

        const Moments moment = moments(convex_contour, true);

        const Point2d image_point(moment.m10 / moment.m00, moment.m01 / moment.m00);
        const cv::Point3d position = engine.estimate_position(image_point, area);

        ball.position = geometry_msgs::msg::Point().set__x(position.x).set__y(position.y).set__z(position.z);
        ball.color = 0U;
        ball.id = 0U;

        message.balls.push_back(ball);

    }
}

Application::CameraCalibration Application::load_calibration_file(){
    CameraCalibration result;

    std::ifstream ifs("./src/image_sensign/src/camera_calibration.json");
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

int main(int argc, char* argv[]){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Application>());
    rclcpp::shutdown();
    return 0;
}