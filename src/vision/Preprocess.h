#pragma once
#include <esp_camera.h>
#include "ModelInference.h"

namespace Preprocess {

// Chép ảnh RGB565 từ camera vào tensor input của model: letterbox (đệm màu xám 114) + lượng tử hóa bằng LUT.
// Trả false nếu buffer input của model nhỏ hơn cần thiết.
bool frameToInput(const camera_fb_t* fb, ModelInference& model);

}  // namespace Preprocess
