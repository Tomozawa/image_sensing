cmake_minimum_required(VERSION 3.27.7)

project(image_sensing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ./include)

add_executable(
    ball_detecter
    ball_detecter.cpp
    utils.cpp
)

target_link_libraries(ball_detecter ${OpenCV_LIBRARIES})