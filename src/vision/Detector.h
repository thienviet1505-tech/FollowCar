#pragma once
#include <vector>
#include <cstdint>
#include "ModelInference.h"

struct Detection {
    int class_id = -1;
    float score = 0.0f;
    int cx = 0, cy = 0;  // tâm box, pixel theo ảnh gốc
    int w = 0, h = 0;    // kích thước box, pixel theo ảnh gốc
};

class Detector {
public:
    // origWidth/origHeight: kích thước ảnh gốc từ camera, để quy đổi box về pixel.
    static void decode(const ModelInference& model, int origWidth, int origHeight,
                        std::vector<Detection>& outDetections);

    // Điểm cao nhất của từng class trong frame vừa decode (tính trước khi lọc
    // ngưỡng) — để biết model "đang nghĩ gì" kể cả khi chưa vượt CONF_THRESHOLD.
    static const std::vector<float>& classMaxScores();

};