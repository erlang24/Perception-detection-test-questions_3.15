#ifndef YOLO_DETECTOR_HPP_
#define YOLO_DETECTOR_HPP_

#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>
#include "object_detection/detection_result.hpp"

class YOLODetector {
public:
    YOLODetector();
    ~YOLODetector();

    std::vector<DetectionResult> detect(const cv::Mat& image);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

#endif  // YOLO_DETECTOR_HPP_ 