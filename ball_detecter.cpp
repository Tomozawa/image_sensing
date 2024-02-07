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
    // CameraMatrix camera_matrix;
    CameraMatrix left_camera_matrix;
    CameraMatrix right_camera_matrix;
    // Distorsion distorsion;
    Distorsion left_distorsion;
    Distorsion right_distorsion;
    RMatrix r_matrix;
    TVec t_vector;
    double focal_distance;
};

// struct CameraScaling{
//     double distance;
//     double area;
// };

// void execute_calc(InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&, const PositionEstimateEngine<EngineType::MONO_CAM>&);
// void execute_calc(InputArray, InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&, const PositionEstimateEngine<EngineType::STEREO_CAM_SGBM>&);
void execute_calc(InputArray, InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&, const PositionEstimateEngine<EngineType::STEREO_CAM_RRM>&);
CameraCalibration load_calibration_file(void);
// CameraScaling load_scaling_file(void);
std::vector<Point2i> find_balls(const Mat, const InRangeParams&, OutputArray);

//グローバル変数を包む構造体
//グローバル変数が必要なのでやむなし
//必要なアクセス以外しないよう、またデバッグがやりやすいようゲッター、セッター、ラッパーで包んでいる
struct GlobalVariables{
    private:
        InRangeParams params_ = {0};
        // VideoCapture video_capture_;
        VideoCapture left_video_capture_;
        VideoCapture right_video_capture_;
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

        /*left_video_capture_*/
        /*right_video_capture_*/
        inline bool set_resolution(const Size2i size){
            left_video_capture_.set(CAP_PROP_FRAME_WIDTH, size.width);
            left_video_capture_.set(CAP_PROP_FRAME_HEIGHT, size.height);
            right_video_capture_.set(CAP_PROP_FRAME_WIDTH, size.width);
            right_video_capture_.set(CAP_PROP_FRAME_HEIGHT, size.height);

            return(
                left_video_capture_.get(CAP_PROP_FRAME_WIDTH) == size.width
                && left_video_capture_.get(CAP_PROP_FRAME_HEIGHT) == size.height
                && right_video_capture_.get(CAP_PROP_FRAME_WIDTH) == size.width
                && right_video_capture_.get(CAP_PROP_FRAME_HEIGHT) == size.height
            );
        }

        inline bool open_cameras(void){
            left_video_capture_.open(1);
            right_video_capture_.open(2);
            return left_video_capture_.isOpened() && right_video_capture_.isOpened();
        }
        
        inline bool grabs(void){return left_video_capture_.grab() && right_video_capture_.grab();}

        // inline bool retrieve(OutputArray output, int flag=0){return left_video_capture_.retrieve(output, flag);}
        inline bool left_retrieve(OutputArray output, int flag=0){return left_video_capture_.retrieve(output, flag);}
        inline bool right_retrieve(OutputArray output, int flag=0){return right_video_capture_.retrieve(output, flag);}

        /*lut*/
        inline LUT_TYPE get_hue_lut(void){return hue_lut_;}
};

GlobalVariables global_variables;

int main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);
#ifdef _DEBUG
    CV_LOG_DEBUG(nullptr, "In debug mode");
#endif

    CameraCalibration calibration;
    // CameraScaling scaling;

    try{
        calibration = load_calibration_file();
        // scaling = load_scaling_file();
    }catch(std::runtime_error& e){
        CV_LOG_ERROR(nullptr, e.what());
        std::exit(-1);
    }catch(json::exception& e){
        CV_LOG_ERROR(nullptr, e.what());
        std::exit(-1);
    }

    // constexpr ScalingFactor scaling_factor = {};
    // const PositionEstimateEngine<EngineType::MONO_CAM> engine(
    //     ScalingFactor{
    //         .area = scaling.area,
    //         .distance = scaling.distance
    //     },
    //     calibration.camera_matrix
    // );

    const Size2i image_size(1280, 720);
    constexpr int num_channels = 3;

    Mat mapl1, mapl2, mapr1, mapr2;
    QMatrix Q;
    {
        Mat Rl, Rr, Pl, Pr;
        cv::stereoRectify(
            calibration.left_camera_matrix,
            calibration.left_distorsion,
            calibration.right_camera_matrix,
            calibration.right_distorsion,
            image_size,
            calibration.r_matrix,
            calibration.t_vector,
            Rl,
            Rr,
            Pl,
            Pr,
            Q,
            0,
            0
        );
        cv::initUndistortRectifyMap(
            calibration.left_camera_matrix,
            calibration.left_distorsion,
            Rl,
            Pl,
            image_size,
            CV_16SC2,
            mapl1,
            mapl2
        );
        cv::initUndistortRectifyMap(
            calibration.right_camera_matrix,
            calibration.right_distorsion,
            Rr,
            Pr,
            image_size,
            CV_16SC2,
            mapr1,
            mapr2
        );
    }

    const PositionEstimateEngine<EngineType::STEREO_CAM_RRM> engine(Q);

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
    CV_Assert(global_variables.set_resolution(image_size));

    do{
        global_variables.grabs();
        // Mat input_img, undistort_img, output1_img, output2_img, output3_img;
        Mat left_input_img, right_input_img, left_recify_img, right_recify_img, output1_img, output2_img, output3_img;
        
        // global_variables.retrieve(input_img);
        global_variables.left_retrieve(left_input_img);
        global_variables.right_retrieve(right_input_img);
        CV_DbgAssert(left_input_img.cols == image_size.width && left_input_img.rows == image_size.height);
        CV_DbgAssert(right_input_img.cols == image_size.width && right_input_img.rows == image_size.height);
        CV_DbgAssert(left_input_img.channels() == num_channels && right_input_img.channels() == num_channels);

        // undistort(input_img, undistort_img, calibration.camera_matrix, calibration.distorsion);
        // undistort(left_input_img, left_undistort_img, calibration.left_camera_matrix, calibration.left_distorsion);
        // undistort(right_input_img, right_undistort_img, calibration.right_camera_matrix, calibration.right_distorsion);
        remap(
            left_input_img,
            left_recify_img,
            mapl1,
            mapl2,
            INTER_LINEAR
        );
        remap(
            right_input_img,
            right_recify_img,
            mapr1,
            mapr2,
            INTER_LINEAR
        );

        // execute_calc(undistort_img, output1_img, output2_img, output3_img, global_variables.replace_param(), engine);
        execute_calc(left_recify_img, right_recify_img, output1_img, output2_img, output3_img, global_variables.replace_param(), engine);

        imshow("output1", output1_img);
        imshow("output2", output2_img);
        imshow("output3", output3_img);
    }while(waitKey(1) == -1);

    CV_LOG_DEBUG(nullptr, "clean up");

    return 0;
}

// void execute_calc(InputArray input, OutputArray output1, OutputArray output2, OutputArray output3, const InRangeParams& params, const PositionEstimateEngine<EngineType::MONO_CAM>& engine){
//     Mat image, hsv, hsv_blur, hsv_filtered, closed, opened, canny_img, image_with_contours;
//     std::vector<std::vector<Point>> contours;
//     std::vector<Vec4i> hierarchy;
//     CV_DbgAssert(input.isMat());
//     image = input.getMat();

//     cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
//     GaussianBlur(hsv, hsv_blur, Size(11, 11), 8.5, 8.5);

//     //普通のinRangeだと赤色が検知できない
//     hsv_range(hsv_blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

//     opening<2>(hsv_filtered, opened);
//     closing<2>(opened, closed);
    
//     Canny(closed, canny_img, 25, 75);

//     findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

//     image.copyTo(image_with_contours);

//     for(int i = 0; i < contours.size(); i++){
//         if(hierarchy.at(i)[3] >= 0) continue;

//         std::vector<Point> applox_contour, convex_contour;
//         approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
//         convexHull(applox_contour, convex_contour);

//         const double area = contourArea(convex_contour);

//         if(area < 400) continue;
//         if(convex_contour.size() < 10) continue;

//         const Moments moment = moments(convex_contour, true);

//         // polylines(image_with_contours, convex_contour, true, Scalar{0, 0, 255});
//         // putText(
//         //     image_with_contours,
//         //     cv::format("%d-gon", convex_contour.size()),
//         //     Point{static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00)},
//         //     HersheyFonts::FONT_HERSHEY_SIMPLEX,
//         //     1,
//         //     Scalar{0, 0, 255},
//         //     3
//         // );

//         const Point2d image_point(moment.m10 / moment.m00, moment.m01 / moment.m00);

//         const cv::Point3d position = engine.estimate_position(image_point, area);

//         polylines(image_with_contours, convex_contour, true, Scalar{0, 0, 255});
//         putText(
//             image_with_contours,
//             cv::format("(%d, %d, %d)", static_cast<int>(position.x), static_cast<int>(position.y), static_cast<int>(position.z)),
//             Point{static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00)},
//             HersheyFonts::FONT_HERSHEY_SIMPLEX,
//             1,
//             Scalar{0, 0, 255},
//             3
//         );
//     }

//     output1.move(hsv_filtered);
//     output2.move(opened);
//     output3.move(image_with_contours);
// }

// void execute_calc(
//     InputArray left_input,
//     InputArray right_input,
//     OutputArray output1,
//     OutputArray output2,
//     OutputArray output3,
//     const InRangeParams& params,
//     const PositionEstimateEngine<EngineType::STEREO_CAM_SGBM>& engine
// ){
//     CV_DbgAssert(left_input.isMat() && right_input.isMat());
//     Mat left_image = left_input.getMat();
//     Mat right_image = right_input.getMat();
//     Mat disp;

//     {
//         Mat left_gray, right_gray, left_blur, right_blur, normalized_disp;

//         cvtColor(
//             left_image,
//             left_gray,
//             COLOR_RGB2GRAY
//         );
//         cvtColor(
//             right_image,
//             right_gray,
//             COLOR_RGB2GRAY
//         );
//         GaussianBlur(left_gray, left_blur, Size(3, 3), 1);
//         GaussianBlur(right_gray, right_blur, Size(3, 3),1);

//         disp = engine.get_disp(left_blur, right_blur);

//         normalize(
//             disp,
//             normalized_disp,
//             0,
//             UINT8_MAX,
//             NORM_MINMAX,
//             CV_8UC1
//         );

//         output3.move(normalized_disp);
//     }

//     {
//         Mat hsv, hsv_blur, hsv_filtered, closed, opened, canny_img, left_image_with_contours;
//         std::vector<std::vector<Point>> contours;
//         std::vector<Vec4i> hierarchy;

//         cvtColor(left_image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
//         GaussianBlur(hsv, hsv_blur, Size(11, 11), 8.5);

//         //普通のinRangeだと赤色が検知できない
//         hsv_range(hsv_blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

//         opening<2>(hsv_filtered, opened);
//         closing<2>(opened, closed);
        
//         Canny(closed, canny_img, 25, 75);

//         findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

//         left_image.copyTo(left_image_with_contours);

//         for(int i = 0; i < contours.size(); i++){
//             if(hierarchy.at(i)[3] >= 0) continue;

//             std::vector<Point> applox_contour, convex_contour;
//             approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
//             convexHull(applox_contour, convex_contour);

//             const double area = contourArea(convex_contour);

//             if(area < 400) continue;
//             if(convex_contour.size() < 10) continue;

//             const Moments moment = moments(convex_contour, true);
//             const Point2d image_point(moment.m10 / moment.m00, moment.m01 / moment.m00);
//             const cv::Point3d position = engine.estimate_position(disp, image_point);

//             polylines(left_image_with_contours, convex_contour, true, Scalar{0, 0, 255});
//             putText(
//                 left_image_with_contours,
//                 /*position*/format("%lf", static_cast<double>(disp.at<short>(image_point.y, image_point.x)) / 16.0),
//                 Point{static_cast<int>(moment.m10 / moment.m00), static_cast<int>(moment.m01 / moment.m00)},
//                 HersheyFonts::FONT_HERSHEY_SIMPLEX,
//                 1,
//                 Scalar{0, 0, 255},
//                 3
//             );
//         }
//     }
//     output1.move(left_image);
//     output2.move(right_image);
// }

void execute_calc(
    InputArray left_input,
    InputArray right_input,
    OutputArray output1,
    OutputArray output2,
    OutputArray output3,
    const InRangeParams& params,
    const PositionEstimateEngine<EngineType::STEREO_CAM_RRM>& engine
){
    CV_DbgAssert(left_input.isMat() && right_input.isMat());
    const Mat left_image = left_input.getMat();
    const Mat right_image = right_input.getMat();

    const std::vector<Point2i> left_balls = find_balls(left_image, params, output1);
    const std::vector<Point2i> right_balls = find_balls(right_image, params, output2);

    const auto finalize = [output3, left_image](){
        left_image.copyTo(output3);
    };

    if(left_balls.size() != right_balls.size()){
        finalize();
        return;
    }

    std::vector<Point3d> positions;
    const auto positions_result = engine.estimate_positions(left_balls, right_balls);
    if(positions_result) positions = positions_result.value();
    else{
        finalize();
        return;
    }

    Mat left_image_with_positions = left_image.clone();

    CV_DbgAssert(left_balls.size() == positions.size());
    for(unsigned i = 0; i < left_balls.size(); ++i){
        putText(
            left_image_with_positions,
            cv::format("(%d, %d, %d)", positions.at(i).x, positions.at(i).y, positions.at(i).z),
            left_balls.at(i),
            FONT_HERSHEY_COMPLEX,
            1,
            Scalar(0, 0, 255),
            3
        );
    }

    output3.move(left_image_with_positions);
}


std::vector<Point2i> find_balls(const Mat input, const InRangeParams& params, OutputArray output){
    Mat hsv, hsv_blur, hsv_filtered, closed, opened, canny_img, image_with_contours;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    std::vector<Point2i> result;

    cvtColor(input, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, hsv_blur, Size(11, 11), 8.5);

    //普通のinRangeだと赤色が検知できない
    hsv_range(hsv_blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

    opening<2>(hsv_filtered, opened);
    closing<2>(opened, closed);
    
    Canny(closed, canny_img, 25, 75);

    findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    input.copyTo(image_with_contours);

    for(int i = 0; i < contours.size(); i++){
        if(hierarchy.at(i)[3] >= 0) continue;

        std::vector<Point> applox_contour, convex_contour;
        approxPolyDP(contours[i], applox_contour, 0.005 * arcLength(contours[i], true), true);
        convexHull(applox_contour, convex_contour);

        const double area = contourArea(convex_contour);

        if(area < 1200) continue;
        if(convex_contour.size() != 4) continue;

        const Moments moment = moments(convex_contour, true);
        result.push_back(Point2i(moment.m10 / moment.m00, moment.m01 / moment.m00));
    }

    output.move(closed);

    return result;
}

CameraCalibration load_calibration_file(){
    CameraCalibration result;

    // std::ifstream ifs("camera_calibration.json");
    std::ifstream ifs("stereo_camera_calibration.json");
    json calibration_json;
    if(!ifs.is_open()) throw std::runtime_error("Can't open stereo_camera_calibration.json");

    ifs >> calibration_json;

    ifs.close();

    CV_LOG_DEBUG(nullptr, "matrix");

    // const json& matrix_ref = calibration_json.at("matrix");
    // result.camera_matrix = {
    //     matrix_ref.at(0).at(0), matrix_ref.at(0).at(1), matrix_ref.at(0).at(2),
    //     matrix_ref.at(1).at(0), matrix_ref.at(1).at(1), matrix_ref.at(1).at(2),
    //     matrix_ref.at(2).at(0), matrix_ref.at(2).at(1), matrix_ref.at(2).at(2)
    // };
    const json& left_matrix_ref = calibration_json.at("left_matrix");
    result.left_camera_matrix = {
        left_matrix_ref.at(0).at(0), left_matrix_ref.at(0).at(1), left_matrix_ref.at(0).at(2),
        left_matrix_ref.at(1).at(0), left_matrix_ref.at(1).at(1), left_matrix_ref.at(1).at(2),
        left_matrix_ref.at(2).at(0), left_matrix_ref.at(2).at(1), left_matrix_ref.at(2).at(2)
    };

    const json& right_matrix_ref = calibration_json.at("right_matrix");
    result.right_camera_matrix = {
        right_matrix_ref.at(0).at(0), right_matrix_ref.at(0).at(1), right_matrix_ref.at(0).at(2),
        right_matrix_ref.at(1).at(0), right_matrix_ref.at(1).at(1), right_matrix_ref.at(1).at(2),
        right_matrix_ref.at(2).at(0), right_matrix_ref.at(2).at(1), right_matrix_ref.at(2).at(2)
    };

    // const json& distortion_ref = calibration_json.at("distortion");
    // result.distorsion = {
    //     distortion_ref.at(0).at(0),
    //     distortion_ref.at(1).at(0),
    //     distortion_ref.at(2).at(0),
    //     distortion_ref.at(3).at(0),
    //     distortion_ref.at(4).at(0)
    // };
    const json& left_distortion_ref = calibration_json.at("left_distortion");
    result.left_distorsion = {
        left_distortion_ref.at(0).at(0),
        left_distortion_ref.at(1).at(0),
        left_distortion_ref.at(2).at(0),
        left_distortion_ref.at(3).at(0),
        left_distortion_ref.at(4).at(0)
    };
    const json& right_distortion_ref = calibration_json.at("right_distortion");
    result.right_distorsion = {
        right_distortion_ref.at(0).at(0),
        right_distortion_ref.at(1).at(0),
        right_distortion_ref.at(2).at(0),
        right_distortion_ref.at(3).at(0),
        right_distortion_ref.at(4).at(0)
    };

    const json& r_matrix_ref = calibration_json.at("R_matrix");
    result.r_matrix = {
        r_matrix_ref.at(0).at(0), r_matrix_ref.at(0).at(1), r_matrix_ref.at(0).at(2),
        r_matrix_ref.at(1).at(0), r_matrix_ref.at(1).at(1), r_matrix_ref.at(1).at(2),
        r_matrix_ref.at(2).at(0), r_matrix_ref.at(2).at(1), r_matrix_ref.at(2).at(2)
    };

    const json& t_vector_ref = calibration_json.at("T_vector");
    result.t_vector = {
        t_vector_ref.at(0).at(0),
        t_vector_ref.at(1).at(0),
        t_vector_ref.at(2).at(0)
    };

    result.focal_distance = calibration_json.at("focal_distance");

    return result;
}

// CameraScaling load_scaling_file(){
//     CameraScaling result;

//     std::ifstream ifs("camera_scaling.json");
//     json scaling_json;
//     if(!ifs.is_open()) throw std::runtime_error("Can't open camera_scaling.json");

//     ifs >> scaling_json;

//     ifs.close();

//     result.area = scaling_json.at("area");
//     result.distance = scaling_json.at("distance");

//     return result;
// }