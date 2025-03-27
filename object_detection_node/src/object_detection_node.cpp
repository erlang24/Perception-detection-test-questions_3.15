#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>
#include "object_detection/detection_result.hpp"
#include "object_detection/yolo_detector.hpp"

class ObjectDetectorNode : public rclcpp::Node
{
public:
    ObjectDetectorNode() : Node("object_detector_node")
    {
        // 初始化YOLOv5检测器
        detector_object = std::make_unique<YOLODetector>();

        // 创建订阅者 /image_raw /camera/image_raw
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10,
            std::bind(&ObjectDetectorNode::image_callback, this, std::placeholders::_1));

        // 创建检测结果发布者
        detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/detection_result", 10);
        
        // 创建可视化结果发布者
        visualization_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/detection_visualization", 10);

        RCLCPP_INFO(this->get_logger(), "Object detector node has been initialized.");
        
        // 初始化类别名称
        class_names_ = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try {
            // 将ROS图像消息转换为OpenCV格式
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat image = cv_ptr->image;
            
            RCLCPP_INFO(this->get_logger(), "Image size: %dx%d", image.cols, image.rows);

            // 调用YOLO检测
            std::vector<DetectionResult> detections = detector_object->detect(image);
            
            RCLCPP_INFO(this->get_logger(), "Found %zu detections", detections.size());

            // 创建检测结果消息
            auto detection_array_msg = std::make_unique<vision_msgs::msg::Detection2DArray>();
            detection_array_msg->header = msg->header;

            // 创建可视化图像
            cv::Mat visualization = image.clone();

            // 将检测结果转换为ROS消息并绘制到可视化图像上
            for (const auto& det : detections) {
                // 检查坐标是否在合理范围内
                if (det.x < 0 || det.y < 0 || det.width <= 0 || det.height <= 0 ||
                    det.x + det.width > image.cols || det.y + det.height > image.rows) {
                    RCLCPP_WARN(this->get_logger(), "Invalid detection bbox: (%.1f,%.1f,%.1f,%.1f)",
                        det.x, det.y, det.width, det.height);
                    continue;
                }

                vision_msgs::msg::Detection2D detection;
                detection.header = msg->header;  // 使用输入图像的时间戳
                detection.bbox.center.x = det.x + det.width / 2.0;
                detection.bbox.center.y = det.y + det.height / 2.0;
                detection.bbox.size_x = det.width;
                detection.bbox.size_y = det.height;
                
                // 添加类别信息
                vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
                hypothesis.hypothesis.class_id = std::to_string(det.class_id);  // 将类别ID转换为字符串
                hypothesis.hypothesis.score = det.confidence;
                detection.results.push_back(hypothesis);

                // 添加检测ID
                detection.id = std::to_string(detections.size());  // 使用检测序号作为ID

                detection_array_msg->detections.push_back(detection);

                // 绘制检测框和标签
                cv::Rect bbox(det.x, det.y, det.width, det.height);
                cv::rectangle(visualization, bbox, cv::Scalar(0, 255, 0), 2);
                
                // 准备标签文本
                std::string label = (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) 
                    ? class_names_[det.class_id] 
                    : "unknown";
                label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
                
                // 绘制标签背景
                int baseline = 0;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(visualization, 
                            cv::Point(det.x, det.y - label_size.height - baseline - 5),
                            cv::Point(det.x + label_size.width, det.y),
                            cv::Scalar(0, 255, 0), -1);
                
                // 绘制标签文本
                cv::putText(visualization, label, cv::Point(det.x, det.y - baseline - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                
                RCLCPP_INFO(this->get_logger(), "Detection: class=%s, conf=%.2f, bbox=(%.1f,%.1f,%.1f,%.1f)",
                    label.c_str(), det.confidence, det.x, det.y, det.width, det.height);
            }

            // 发布检测结果
            detection_publisher_->publish(std::move(detection_array_msg));
            
            // 发布可视化结果
            sensor_msgs::msg::Image::SharedPtr visualization_msg = 
                cv_bridge::CvImage(msg->header, "bgr8", visualization).toImageMsg();
            visualization_publisher_->publish(*visualization_msg);
            
            RCLCPP_INFO(this->get_logger(), "Published detection results and visualization");
        }
        catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
            return;
        }
    }

    std::unique_ptr<YOLODetector> detector_object;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visualization_publisher_;
    std::vector<std::string> class_names_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
} 