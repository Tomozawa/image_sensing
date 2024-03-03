cmake_minimum_required(VERSION 3.8)
project(image_sensing)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nhk24_utils REQUIRED)
find_package(backward_ros REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(common_compile_features c_std_17;cxx_std_20)

add_executable(ball_detecter
  src/ball_detecter.cpp
  src/utils.cpp
)
target_include_directories(ball_detecter PUBLIC ./include/image_sensing)
ament_target_dependencies(ball_detecter rclcpp rcpputils OpenCV nhk24_utils)
target_compile_features(ball_detecter PUBLIC ${common_compile_features})

add_executable(setup_filter
  src/setup_filter.cpp
  src/utils.cpp
)
target_include_directories(setup_filter PUBLIC ./include/image_sensing)
ament_target_dependencies(setup_filter PUBLIC rclcpp rcpputils OpenCV nhk24_utils)
target_compile_features(setup_filter PUBLIC ${common_compile_features})

add_backward(ball_detecter)

install(
  TARGETS ball_detecter setup_filter
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
