cmake_minimum_required(VERSION 3.8)
project(object_detection_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可见性标志
add_compile_options(-fvisibility=hidden)

# 设置ONNX Runtime路径
set(ONNXRUNTIME_ROOT "/home/erlang/study_project/kaoshi_3.15/onnxruntime-linux-x64-1.16.3")
if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNX Runtime directory not found. Please run:\ncd /home/erlang/study_project/kaoshi_3.15 && wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && tar xvf onnxruntime-linux-x64-1.16.3.tgz")
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# 添加头文件目录
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(object_detection_node
  src/object_detection_node.cpp
  src/yolo_detector.cpp
)

add_executable(cluster_node
  src/cluster_node.cpp
)

# 指定目标包含目录
target_include_directories(object_detection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${ONNXRUNTIME_ROOT}/include
)

target_include_directories(cluster_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# 链接依赖库
ament_target_dependencies(object_detection_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  cv_bridge
  OpenCV
)

ament_target_dependencies(cluster_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  pcl_conversions
  Eigen3
)

target_link_libraries(object_detection_node
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${Python3_LIBRARIES}
  ${pybind11_LIBRARIES}
)

target_link_libraries(cluster_node
  ${PCL_LIBRARIES}
)

# 安装C++可执行文件
install(TARGETS
  object_detection_node
  cluster_node
  DESTINATION lib/${PROJECT_NAME}
)

# 创建include目录
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# 安装launch文件
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# 安装配置文件
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package() 