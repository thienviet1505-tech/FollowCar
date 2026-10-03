#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel LOGISTIC (sigmoid) int8 dùng bảng tra 256 phần tử, thay cho kernel
// điểm-cố-định của TFLM (rất chậm trên ESP32-S3: SiLU = Logistic + Mul).
// Chỉ hỗ trợ tensor int8. Dùng: resolver.AddLogistic(RegisterLogisticLut());
TFLMRegistration RegisterLogisticLut();
