cmake_minimum_required(VERSION 3.27.7)

project(image_sensing)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ./include)

add_executable(
    silo_filling
    silo_filling.cpp
    utils.cpp
)

target_link_libraries(silo_filling ${OpenCV_LIBRARIES})