#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <cassert>

using namespace cv;

struct InRangeParams{
    typedef int ValueType;
    ValueType h_min;
    ValueType s_min;
    ValueType v_min;
    ValueType h_max;
    ValueType s_max;
    ValueType v_max;
};

void execute_calc(InputArray, OutputArray, OutputArray, OutputArray, const InRangeParams&);

template<unsigned char DilateErodeTimes>
void opening(InputArray, OutputArray);

template<unsigned char DilateErodeTims>
void closing(InputArray, OutputArray);

//グローバル変数を包む構造体
//グローバル変数が必要なのでやむなし
//必要なアクセス以外しないよう、またデバッグがやりやすいようゲッター、セッター、ラッパーで包んでいる
//無名構造体にすることで1つしか作らないでねという意思表示をした(staticだと初期化まわりが大変らしい)(decltypeでインスタンスは作れてしまう)
struct {
    private:
        InRangeParams params_ = {0};
        VideoCapture video_capture_;
    public:
        /* params_ */
        //Todo: 値チェック
        //Todo: インデックスをテンプレート化または列挙子にする
        //取得と設定を同時に行うことを強制する(取得のみすることもできる)
        //取得(ゲット)→更新→更新の適用(セット)を呼び出し元に任せると、更新の適用忘れがあるかもしれないので
        inline InRangeParams replace_param(const InRangeParams::ValueType val, int8_t index){
            switch(index){
                case 0: params_.h_min = val; break;
                case 1: params_.h_max = val; break;
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
} grobal_variables;

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
            grobal_variables.replace_param(val, 0);
        }
    );
    createTrackbar(
        "Hue max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            grobal_variables.replace_param(val, 1);
        }
    );
    createTrackbar(
        "Saturation min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            grobal_variables.replace_param(val, 2);
        }
    );
    createTrackbar(
        "Saturation max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            grobal_variables.replace_param(val, 3);
        }
    );
    createTrackbar(
        "Value min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            grobal_variables.replace_param(val, 4);
        }
    );
    createTrackbar(
        "Value max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            grobal_variables.replace_param(val, 5);
        }
    );

    CV_Assert(grobal_variables.open_cameras());

    do{
        grobal_variables.grabs();
        Mat input_img, output1_img, output2_img, output3_img;
        grobal_variables.retrieve(input_img);

        execute_calc(input_img, output1_img, output2_img, output3_img, grobal_variables.replace_param());

        imshow("output1", output1_img);
        imshow("output2", output2_img);
        imshow("output3", output3_img);
    }while(waitKey(1) == -1);

    CV_LOG_DEBUG(nullptr, "clean up");

    return 0;
}

void execute_calc(InputArray input, OutputArray output1, OutputArray output2, OutputArray output3, const InRangeParams& params){
    Mat image, hsv, blur, hsv_filtered, morph, canny_img, image_with_contours;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    CV_DbgAssert(input.isMat());
    image = input.getMat();

    cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(7, 7), 5, 5);
    Scalar lowerb(params.h_min, params.s_min, params.v_min), upperb(params.h_max, params.s_max, params.v_max);
    inRange(blur, lowerb, upperb, hsv_filtered);
    
    Canny(hsv_filtered, canny_img, 25, 75);
    closing<1>(canny_img, morph);

    findContours(morph, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    image.copyTo(image_with_contours);

    for(int i = 0; i < contours.size(); i++){
        if(contourArea(contours[i]) < 400) continue;
        drawContours(image_with_contours, contours, i, Scalar(0, 0, 255));
    }
    output1.move(hsv_filtered);
    output2.move(morph);
    output3.move(image_with_contours);
}

#define BUF_TO_WRITE (mat_buf[buf_index])
#define BUF_TO_READ (mat_buf[(buf_index + sizeof(mat_buf) / sizeof(mat_buf[0]) - 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]))])
#define SWTICH_BUF() do{\
    buf_index = (buf_index + 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]));\
}while(0)\

template<unsigned char DilateErodeTimes>
void closing(InputArray input, OutputArray output){
    
    CV_DbgAssert(input.isMat());

    Mat mat_buf[2] = {Mat(), Mat(input.getMat())};
    unsigned char buf_index = 0;
    Mat kernel = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));

    //closing
    dilate(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    erode(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();

    //opening
    erode(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    dilate(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();

    output.move(BUF_TO_READ);
}

#undef BUF_TO_WRITE
#undef BUF_TO_READ

#define BUF_TO_WRITE (mat_buf[buf_index])
#define BUF_TO_READ (mat_buf[(buf_index + sizeof(mat_buf) / sizeof(mat_buf[0]) - 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]))])
#define SWTICH_BUF() do{\
    buf_index = (buf_index + 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]));\
}while(0)\

template<unsigned char DilateErodeTimes>
void opening(InputArray input, OutputArray output){
    
    CV_DbgAssert(input.isMat());

    Mat mat_buf[2] = {Mat(), Mat(input.getMat())};
    unsigned char buf_index = 0;
    Mat kernel = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));

    //opening
    erode(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    dilate(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();

    output.move(BUF_TO_READ);
}

#undef BUF_TO_WRITE
#undef BUF_TO_READ