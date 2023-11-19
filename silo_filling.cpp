#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <InRangeParams.hpp>
#include <HueLookupTable.hpp>
#include <utils.hpp>

using namespace cv;
using namespace in_range_params;
using namespace sensing_utils;
using namespace hue_lut;

void execute_calc(InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&);

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
            video_capture_.open(0);
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
        Mat input_img, output1_img, output2_img, output3_img;
        global_variables.retrieve(input_img);

        execute_calc(input_img, output1_img, output2_img, output3_img, global_variables.replace_param());

        imshow("output1", output1_img);
        imshow("output2", output2_img);
        imshow("output3", output3_img);
    }while(waitKey(1) == -1);

    CV_LOG_DEBUG(nullptr, "clean up");

    return 0;
}

void execute_calc(InputArray input, OutputArray output1, OutputArray output2, OutputArray output3, const InRangeParams& params){
    Mat image, hsv, blur, hsv_filtered, closed, opened, canny_img, image_with_contours;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    CV_DbgAssert(input.isMat());
    image = input.getMat();

    cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(11, 11), 8.5, 8.5);

    //普通のinRangeだと赤色が検知できない
    hsv_range(blur, global_variables.get_hue_lut(), params.s_min, params.s_max, params.v_min, params.v_max, hsv_filtered);

    opening<1>(hsv_filtered, opened);
    closing<1>(opened, closed);
    
    Canny(closed, canny_img, 25, 75);

    findContours(canny_img, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    image.copyTo(image_with_contours);

    for(int i = 0; i < contours.size(); i++){
        if(hierarchy.at(i)[3] >= 0) continue;

        std::vector<Point> applox_contour, convex_contour;
        approxPolyDP(contours[i], applox_contour, 0.01 * arcLength(contours[i], true), true);
        convexHull(applox_contour, convex_contour);

        const double area = contourArea(convex_contour);

        if(area < 400) continue;
        //if(convex_contour.size() < 10) continue;

        const Moments moment = moments(convex_contour, true);

        polylines(image_with_contours, convex_contour, true, Scalar{0, 0, 255});
        putText(
            image_with_contours,
            cv::format("%d-gon", convex_contour.size()),
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