#pragma once
#include <algorithm>
#include <cmath>

// Phép letterbox ảnh camera (srcW x srcH) -> khung input model (modelW x modelH): thu nhỏ giữ tỉ lệ rồi đệm hai bên.
// Dùng chung cho tiền xử lý (Preprocess.cpp) và đổi box về ảnh gốc (Detector.cpp) để hai nơi không bao giờ lệch nhau.
struct Letterbox {
    float scale = 1.0f;       // model / camera
    int resizeW = 0, resizeH = 0;   // kích thước ảnh sau khi thu nhỏ (chưa đệm)
    int padX = 0, padY = 0;         // lề trái / lề trên trong khung model

    static Letterbox make(int srcW, int srcH, int modelW, int modelH) {
        Letterbox lb;
        const float scaleW = (float)modelW / (float)srcW;
        const float scaleH = (float)modelH / (float)srcH;
        lb.scale = std::min(scaleW, scaleH);
        lb.resizeW = (int)std::round(srcW * lb.scale);
        lb.resizeH = (int)std::round(srcH * lb.scale);
        lb.padX = (int)std::round((modelW - lb.resizeW) / 2.0f - 0.1f);
        lb.padY = (int)std::round((modelH - lb.resizeH) / 2.0f - 0.1f);
        return lb;
    }
};
