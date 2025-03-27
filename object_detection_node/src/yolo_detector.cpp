#include "object_detection/yolo_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>

class YOLODetector::Impl {
public:
    Impl() {
        try {
            // 创建ONNX Runtime环境
            env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolov5");
            session_options = Ort::SessionOptions();
            
            // 设置线程数和优化级别
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 加载模型
            const char* model_path = "/home/erlang/study_project/kaoshi_3.15/yolov5/weights/yolov5s.onnx";
            session = Ort::Session(env, model_path, session_options);
            
            // 获取模型输入信息
            Ort::AllocatorWithDefaultOptions allocator;
            
            // 获取输入名称
            size_t num_input_nodes = session.GetInputCount();
            input_names.reserve(num_input_nodes);
            input_node_names.reserve(num_input_nodes);
            
            for (size_t i = 0; i < num_input_nodes; i++) {
                input_names.push_back(session.GetInputNameAllocated(i, allocator));
                input_node_names.push_back(input_names.back().get());
            }
            
            // 获取输出名称
            size_t num_output_nodes = session.GetOutputCount();
            output_names.reserve(num_output_nodes);
            output_node_names.reserve(num_output_nodes);
            
            for (size_t i = 0; i < num_output_nodes; i++) {
                output_names.push_back(session.GetOutputNameAllocated(i, allocator));
                output_node_names.push_back(output_names.back().get());
            }
            
            // 获取输入维度
            auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            input_height = input_shape[2];
            input_width = input_shape[3];
            
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Failed to initialize ONNX Runtime: " + std::string(e.what()));
        }
    }

    std::vector<DetectionResult> run_inference(const cv::Mat& image) {
        try {
            // 预处理图像
            cv::Mat resized;
            cv::resize(image, resized, cv::Size(input_width, input_height));
            cv::Mat float_img;
            resized.convertTo(float_img, CV_32F, 1.0/255.0);
            
            // 准备输入tensor
            std::vector<float> input_tensor_values(input_height * input_width * 3);
            float* input_ptr = input_tensor_values.data();
            
            // HWC -> CHW
            for(int c = 0; c < 3; c++) {
                for(int h = 0; h < input_height; h++) {
                    for(int w = 0; w < input_width; w++) {
                        input_ptr[c * input_height * input_width + h * input_width + w] = 
                            float_img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            // 创建输入tensor
            std::vector<int64_t> input_shape = {1, 3, input_height, input_width};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size());
            
            // 运行推理
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_node_names.data(),
                &input_tensor,
                1,
                output_node_names.data(),
                1);
            
            // 处理输出
            std::vector<DetectionResult> results;
            if (!output_tensors.empty()) {
                const float* output_data = output_tensors[0].GetTensorData<float>();
                const auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                const int num_detections = output_shape[1];
                const int num_classes = output_shape[2] - 5;  // 减去4个bbox坐标和1个objectness score
                
                // 存储所有检测结果
                std::vector<DetectionResult> all_detections;
                
                for (int i = 0; i < num_detections; i++) {
                    const float* detection = output_data + i * (num_classes + 5);
                    float confidence = detection[4];
                    
                    if (confidence > 0.5f) {  // 置信度阈值
                        // 找到最高置信度的类别
                        int class_id = -1;
                        float max_class_score = 0.0f;
                        for (int j = 0; j < num_classes; j++) {
                            float class_score = detection[5 + j];
                            if (class_score > max_class_score) {
                                max_class_score = class_score;
                                class_id = j;
                            }
                        }
                        
                        if (max_class_score > 0.5f) {  // 类别置信度阈值
                            DetectionResult det;
                            // YOLO输出的是相对于模型输入尺寸的坐标
                            float x_center = detection[0];
                            float y_center = detection[1];
                            float width = detection[2];
                            float height = detection[3];
                            
                            // 计算缩放比例
                            float scale_x = static_cast<float>(image.cols) / static_cast<float>(input_width);
                            float scale_y = static_cast<float>(image.rows) / static_cast<float>(input_height);
                            
                            // 将坐标转换回原始图像尺寸
                            x_center *= scale_x;
                            y_center *= scale_y;
                            width *= scale_x;
                            height *= scale_y;
                            
                            // 计算左上角坐标
                            det.x = x_center - (width / 2.0f);
                            det.y = y_center - (height / 2.0f);
                            det.width = width;
                            det.height = height;
                            
                            // 确保边界框在图像范围内
                            det.x = std::max(0.0f, std::min(det.x, static_cast<float>(image.cols) - det.width));
                            det.y = std::max(0.0f, std::min(det.y, static_cast<float>(image.rows) - det.height));
                            det.width = std::min(det.width, static_cast<float>(image.cols) - det.x);
                            det.height = std::min(det.height, static_cast<float>(image.rows) - det.y);
                            
                            // 只有当边界框有效时才添加检测结果
                            if (det.width > 0 && det.height > 0) {
                                det.confidence = confidence * max_class_score;
                                det.class_id = class_id;
                                all_detections.push_back(det);
                                
                                // // 添加调试输出
                                // std::cout << "Converted detection: x=" << det.x 
                                //           << ", y=" << det.y 
                                //           << ", width=" << det.width 
                                //           << ", height=" << det.height 
                                //           << ", conf=" << det.confidence 
                                //           << ", class=" << det.class_id 
                                //           << ", scale_x=" << scale_x 
                                //           << ", scale_y=" << scale_y << std::endl;
                            }
                        }
                    }
                }
                
                // 非极大值抑制 (NMS)
                std::vector<bool> keep(all_detections.size(), true);
                const float nms_threshold = 0.45f;
                
                for (size_t i = 0; i < all_detections.size(); i++) {
                    if (!keep[i]) continue;
                    
                    for (size_t j = i + 1; j < all_detections.size(); j++) {
                        if (!keep[j]) continue;
                        
                        // 计算IoU
                        float intersection_x1 = std::max(all_detections[i].x, all_detections[j].x);
                        float intersection_y1 = std::max(all_detections[i].y, all_detections[j].y);
                        float intersection_x2 = std::min(all_detections[i].x + all_detections[i].width,
                                                       all_detections[j].x + all_detections[j].width);
                        float intersection_y2 = std::min(all_detections[i].y + all_detections[i].height,
                                                       all_detections[j].y + all_detections[j].height);
                        
                        if (intersection_x1 < intersection_x2 && intersection_y1 < intersection_y2) {
                            float intersection_area = (intersection_x2 - intersection_x1) * 
                                                    (intersection_y2 - intersection_y1);
                            float union_area = all_detections[i].width * all_detections[i].height +
                                             all_detections[j].width * all_detections[j].height -
                                             intersection_area;
                            float iou = intersection_area / union_area;
                            
                            if (iou > nms_threshold) {
                                if (all_detections[i].confidence > all_detections[j].confidence) {
                                    keep[j] = false;
                                } else {
                                    keep[i] = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // 保存通过NMS的检测结果
                for (size_t i = 0; i < all_detections.size(); i++) {
                    if (keep[i]) {
                        results.push_back(all_detections[i]);
                    }
                }
            }
            
            return results;
            
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Error during inference: " + std::string(e.what()));
        }
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    
    std::vector<Ort::AllocatedStringPtr> input_names;
    std::vector<const char*> input_node_names;
    std::vector<Ort::AllocatedStringPtr> output_names;
    std::vector<const char*> output_node_names;
    
    int input_height;
    int input_width;
};

YOLODetector::YOLODetector() : pimpl_(std::make_unique<Impl>()) {}
YOLODetector::~YOLODetector() = default;

std::vector<DetectionResult> YOLODetector::detect(const cv::Mat& image) {
    // std::cout<<"******image.cols:"<<image.cols<<std::endl;
    // std::cout<<"******image.rows:"<<image.rows<<std::endl;
    return pimpl_->run_inference(image);
} 
