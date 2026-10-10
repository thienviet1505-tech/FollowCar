#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel TRANSPOSE int8 (tối đa 6 chiều): gộp các chiều liền nhau, copy theo khối khi có thể,
// có đường riêng cho NCHW -> NHWC ít kênh (phép đổi đầu model). Kiểu khác int8 chạy kernel gốc.
// Dùng: resolver.AddTranspose(RegisterTransposeFast());
TFLMRegistration RegisterTransposeFast();
