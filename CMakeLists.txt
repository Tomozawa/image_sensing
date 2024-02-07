cmake_minimum_required(VERSION 3.27.7)

project(image_sensing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ./include)

add_executable(
    ball_detecter
    ball_detecter.cpp
    utils.cpp
)

target_link_libraries(ball_detecter ${OpenCV_LIBRARIES})

add_executable(
    StereoCamEstimateEngineTest
    StereoCamEstimateEngineTest.cpp
)

target_link_libraries(StereoCamEstimateEngineTest ${OpenCV_LIBRARIES})

add_executable(
    StereoRRMTest
    StereoRRMTest.cpp
)

target_link_libraries(StereoRRMTest ${OpenCV_LIBRARIES})