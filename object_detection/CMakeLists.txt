cmake_minimum_required(VERSION 3.8)
project(object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(object_detection_interfaces REQUIRED)
find_package(perception_pcl REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL 1.10 REQUIRED)

include_directories(SYSTEM 
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  include
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## IP CAMERA PUBLISHER
add_executable(ip_camera_publisher src/ip_camera_publisher.cpp)

ament_target_dependencies(ip_camera_publisher
  rclcpp
  sensor_msgs
  cv_bridge)

target_link_libraries(ip_camera_publisher
  ament_index_cpp::ament_index_cpp
  ${OpenCV_LIBRARIES}
  nlohmann_json::nlohmann_json)

## POINTCLOUD FILTER SERVICE SERVER
add_executable(pointcloud_filter_srv
  src/pointcloud_filter_node.cpp src/pointcloud_filter_srv.cpp)
target_link_libraries(pointcloud_filter_srv
  # ${Boost_SYSTEM_LIBRARY}
  ${PCL_LIBRARIES}
)
ament_target_dependencies(pointcloud_filter_srv
  rclcpp sensor_msgs pcl_conversions PCL object_detection_interfaces)

install(TARGETS
  ip_camera_publisher
  pointcloud_filter_srv
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_python_install_package(${PROJECT_NAME})
 
install(PROGRAMS
  scripts/yolov8_ip_camera.py
  scripts/yolov8_realsense.py
  scripts/yolov8_luci.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  doors_and_tables.pt
  config.json
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
