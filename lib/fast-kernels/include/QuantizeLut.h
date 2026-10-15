#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel QUANTIZE int8 -> int8 (requantize) dùng bảng tra 256 phần tử.
// Chỉ hỗ trợ input và output đều int8. Dùng: resolver.AddQuantize(RegisterQuantizeLut());
TFLMRegistration RegisterQuantizeLut();
