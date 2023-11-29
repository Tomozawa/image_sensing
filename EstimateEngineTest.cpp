#include <opencv2/core/mat.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <PositionEstimateEngine.hpp>

using namespace cv;
using namespace position_estimate_engine;

int main(){
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_DEBUG);
    const ScalingFactor scaling{
        .area = 1520.0,
        .distance = 1
    };
    const Matx<double, 3, 3> camera_matrix(
        751.5, 0, 400, 0, 845.4, 300, 0, 0, 1
    );
    const PositionEstimateEngine<EngineType::MONO_CAM> engine(scaling, camera_matrix);

    const cv::Point3d obj_vec = engine.estimate_position(Point2d(400, 600), 1520);
    CV_LOG_DEBUG(nullptr, obj_vec);
}