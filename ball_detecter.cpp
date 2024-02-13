#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <fstream>
#include <exception>
#include <string>
#include <vector>
#include <PositionEstimateEngine.hpp>
#include <InRangeParams.hpp>
#include <HueLookupTable.hpp>
#include <utils.hpp>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace in_range_params;
using namespace sensing_utils;
using namespace hue_lut;
using namespace position_estimate_engine;
using namespace nlohmann;

struct CameraCalibration{
    CameraMatrix camera_matrix;
    Vec<double, 5> distorsion;
    double focal_distance;
    Size image_size;
};

void execute_calc(InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&, const PositionEstimateEngine<EngineType::MONO_CAM>&);
CameraCalibration load_calibration_file(void);

//グローバル変数を包む構造体
//グローバル変数が必要なのでやむなし
//必要なアクセス以外しないよう、またデバッグがやりやすいようゲッター、セッター、ラッパーで包んでいる
//無名構造体にすることで1つしか作らないでねという意思表示をした(staticだと初期化まわりが大変らしい)(decltypeでインスタンスは作れてしまう)
struct GlobalVariables{
    private:
        InRangeParams params_ = {0};
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

int main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);
    CameraCalibration calibration;

    try{
        calibration = load_calibration_file();
    }catch(std::runtime_error& e){
        CV_LOG_ERROR(nullptr, e.what());
        std::exit(-1);
    }catch(json::exception& e){
        CV_LOG_ERROR(nullptr, e.what());
        std::exit(-1);
    }

    const PositionEstimateEngine<EngineType::MONO_CAM> engine(
        PointingFactor{
            .focal_distance = calibration.focal_distance,
            .image_size = calibration.image_size
        },
        calibration.camera_matrix
    );

    const String window_name = "Canny";
    namedWindow(window_name);
    createTrackbar(
        "Hue min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 0);
        }
    );
    createTrackbar(
        "Hue max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 1);
        }
    );
    createTrackbar(
        "Saturation min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 2);
        }
    );
    createTrackbar(
        "Saturation max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 3);
        }
    );
    createTrackbar(
        "Value min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 4);
        }
    );
    createTrackbar(
        "Value max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            global_variables.replace_param(val, 5);
        }
    );

    CV_Assert(global_variables.open_cameras());

    do{
        global_variables.grabs();
        Mat input_img, undistort_img, output1_img, output2_img, output3_img;
        global_variables.retrieve(input_img);

        undistort(input_img, undistort_img, calibration.camera_matrix, calibration.distorsion);

        execute_calc(undistort_img, output1_img, output2_img, output3_img, global_variables.replace_param(), engine);

        imshow("output1", output1_img);
        imshow("output2", output2_img);
        imshow("output3", output3_img);
    }while(waitKey(1) == -1);

    CV_LOG_DEBUG(nullptr, "clean up");

    return 0;
}

void execute_calc(InputArray input, OutputArray output1, OutputArray output2, OutputArray output3, const InRangeParams& params, const PositionEstimateEngine<EngineType::MONO_CAM>& engine){
    Mat image, hsv, blur, hsv_filtered, closed, opened, canny_img, image_with_contours;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    CV_DbgAssert(input.isMat());
    image = input.getMat();

    cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(11, 11), 8.5, 8.5);

    //普通のinRangeだと赤色が検知できない
    hsv_range(blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

    opening<2>(hsv_filtered, opened);
    closing<2>(opened, closed);
    
    Canny(closed, canny_img, 25, 75);

    findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    image.copyTo(image_with_contours);

    for(int i = 0; i < contours.size(); i++){
        if(hierarchy.at(i)[3] >= 0) continue;

        std::vector<Point> applox_contour, convex_contour;
        approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
        convexHull(applox_contour, convex_contour);

        const double area = contourArea(convex_contour);

        if(area < 400) continue;
        if(convex_contour.size() < 10) continue;

        const Moments moment = moments(convex_contour, true);

        // polylines(image_with_contours, convex_contour, true, Scalar{0, 0, 255});
        // putText(
        //     image_with_contours,
        //     cv::format("%d-gon", convex_contour.size()),
        //     Point{static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00)},
        //     HersheyFonts::FONT_HERSHEY_SIMPLEX,
        //     1,
        //     Scalar{0, 0, 255},
        //     3
        // );

        const Point2d image_point(moment.m10 / moment.m00, moment.m01 / moment.m00);
        const cv::Point3d position = engine.estimate_position(image_point, area);

        polylines(image_with_contours, convex_contour, true, Scalar{0, 0, 255});
        putText(
            image_with_contours,
            cv::format("(%d, %d, %d, %d)", static_cast<int>(position.x), static_cast<int>(position.y), static_cast<int>(position.z), static_cast<int>(cv::norm(position))),
            Point{static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00)},
            HersheyFonts::FONT_HERSHEY_SIMPLEX,
            1,
            Scalar{0, 0, 255},
            3
        );
    }

    output1.move(hsv_filtered);
    output2.move(opened);
    output3.move(image_with_contours);
}

CameraCalibration load_calibration_file(){
    CameraCalibration result;

    std::ifstream ifs("camera_calibration.json");
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