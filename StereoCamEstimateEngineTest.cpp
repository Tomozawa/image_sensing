#include <PositionEstimateEngine.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace position_estimate_engine;
using namespace cv;

int main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);

    const CameraMatrix left_camera_matrix(
        853.79, 0, 338.74,
        0, 847.85, 348.28,
        0, 0, 1
    );
    const Distorsion left_distorsion(
        -0.13593,
        2.1013,
        0.0010071,
        0.0015141,
        -8.9934
    );
    const CameraMatrix right_camera_matrix(
        866.82, 0, 368.71,
        0, 861.47, 359.33,
        0, 0, 1
    );
    const RMatrix r(
        0.99962, -0.011194, 0.025112,
        0.011063, 0.99992, 0.0053205,
        -0.025170, -0.0050406, 0.99967
    );
    const TVec t(
        13.095,
        -3.1320,
        7.9749
    );
    Mat lsrc, rsrc, lblur, rblur, disp;
    lsrc = imread("lcalibration.jpg");
    rsrc = imread("rcalibration.jpg");

    CV_DbgAssert(lsrc.data && rsrc.data);
    CV_DbgAssert(lsrc.cols == rsrc.cols && lsrc.rows == rsrc.rows);
    CV_DbgAssert(lsrc.channels() == rsrc.channels());

    const auto engine = PositionEstimateEngine<EngineType::STEREO_CAM>::create(
        Size2i(
            lsrc.cols,
            lsrc.rows
        ),
        lsrc.channels(),
        left_camera_matrix,
        right_camera_matrix,
        r,
        t
    );

    GaussianBlur(lsrc, lblur, Size(3, 3), 0.8, 0.8);
    GaussianBlur(rsrc, rblur, Size(3, 3), 0.8, 0.8);

    disp = engine.get_disp(lblur, rblur);

    CV_LOG_DEBUG(nullptr, engine.estimate_position(disp, Point2i(lsrc.cols / 2, rsrc.rows / 2)));

    imshow("StereoCamEstimateEngineTest", disp);

    waitKey(0);
}