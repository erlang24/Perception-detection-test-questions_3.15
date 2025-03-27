#ifndef DETECTION_RESULT_HPP
#define DETECTION_RESULT_HPP

#include <string>

struct DetectionResult {
    float x;           // 边界框左上角x坐标
    float y;           // 边界框左上角y坐标
    float width;       // 边界框宽度
    float height;      // 边界框高度
    int class_id;      // 类别ID
    float confidence;  // 置信度
    std::string class_name; // 类别名称
};

#endif // DETECTION_RESULT_HPP 