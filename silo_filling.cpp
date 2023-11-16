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
    int h_min;
    int s_min;
    int v_min;
    int h_max;
    int s_max;
    int v_max;
};

void execute_calc(InputArray, OutputArray, const InRangeParams&);

template<int DilateErodeTimes>
void closing_opening(InputArray, OutputArray);

//グローバル変数を作りたくないのでJava方式で実行
class Application final{
    private:
        static Mat image;
        static Mat canny_img;
        static InRangeParams params;

    public:
        Application() = delete;
        Application(const Application&) = delete;
        Application(Application&&) = delete;
        Application& operator=(const Application&) = delete;
        Application& operator=(Application&&) = delete;
        ~Application() = delete;

        static bool load_image(const cv::String&);

        static int main();
};

Mat Application::image;
Mat Application::canny_img;
InRangeParams Application::params;

bool Application::load_image(const cv::String& file_name){
    Application::image = imread(file_name);
    return image.data;
}

int Application::main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);

    const String window_name = "Canny";
    namedWindow(window_name);
    createTrackbar(
        "Hue min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.h_min = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );
    createTrackbar(
        "Hue max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.h_max = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );
    createTrackbar(
        "Saturation min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.s_min = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );
    createTrackbar(
        "Saturation max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.s_max = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );
    createTrackbar(
        "Value min",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.v_min = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );
    createTrackbar(
        "Value max",
        window_name,
        nullptr,
        255,
        [](int val, void*){
            Application::params.v_max = val;
            execute_calc(Application::image, Application::canny_img, Application::params);
        }
    );

    execute_calc(Application::image, Application::canny_img, Application::params);

    imshow(window_name, Application::canny_img);

    while(waitKey(1) == -1){
        imshow(window_name, Application::canny_img);
    }

    CV_LOG_DEBUG(nullptr, "clean up");

    return 0;
}

int main(){
    Application::load_image("resourse/ball.png");
    return Application::main();
}

void execute_calc(InputArray input, OutputArray output, const InRangeParams& params){
    Mat image, hsv, blur, hsv_filtered, morph, canny_img, image_with_contours;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    CV_DbgAssert(input.isMat());
    image = input.getMat();

    cvtColor(image, hsv, ColorConversionCodes::COLOR_BGR2HSV_FULL);
    GaussianBlur(hsv, blur, Size(5, 5), 0, 0);
    Scalar lowerb(params.h_min, params.s_min, params.v_min), upperb(params.h_max, params.s_max, params.v_max);
    inRange(blur, lowerb, upperb, hsv_filtered);
    
    Canny(hsv_filtered, canny_img, 25, 75);
    closing_opening<1>(canny_img/*hsv_filtered*/, morph);

    findContours(morph, contours, hierarchy, RetrievalModes::RETR_LIST, ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    image.copyTo(image_with_contours);

    for(int i = 0; i < contours.size(); i++){
        if(contourArea(contours[i]) < 400) continue;
        drawContours(image_with_contours, contours, i, Scalar(0, 0, 255));
    }
    output.move(image_with_contours);
}

#define BUF_TO_WRITE (mat_buf[buf_index])
#define BUF_TO_READ (mat_buf[(buf_index + sizeof(mat_buf) / sizeof(mat_buf[0]) - 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]))])
#define SWTICH_BUF() do{\
    buf_index = (buf_index + 1) % (sizeof(mat_buf) / sizeof(mat_buf[0]));\
}while(0)\

template<int DilateErodeTimes>
void closing_opening(InputArray input, OutputArray output){
    
    CV_DbgAssert(input.isMat());

    Mat mat_buf[2] = {Mat(), Mat(input.getMat())};
    unsigned char buf_index = 0;
    Mat kernel = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));

    CV_LOG_DEBUG(nullptr, cv::format("%s", "Start closing opening"));
    CV_LOG_DEBUG(nullptr, (buf_index + sizeof(mat_buf) / sizeof(mat_buf[0]) - 1) % (sizeof(mat_buf) / sizeof(mat_buf[0])));

    //closing
    dilate(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    erode(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    /*
    //opening
    erode(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    dilate(BUF_TO_READ, BUF_TO_WRITE, kernel, Point(-1, -1), DilateErodeTimes);
    SWTICH_BUF();
    */

    output.move(BUF_TO_READ);
}

#undef BUF_TO_WRITE
#undef BUF_TO_READ