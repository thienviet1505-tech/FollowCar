#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel SOFTMAX int8 -> int8 (trục cuối) dùng bảng exp 256 phần tử và phép tính float, thay cho
// kernel điểm-cố-định của TFLM (rất chậm trên ESP32-S3). Kiểu khác chạy kernel gốc.
// Dùng: resolver.AddSoftmax(RegisterSoftmaxLut());
TFLMRegistration RegisterSoftmaxLut();
