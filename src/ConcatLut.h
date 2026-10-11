#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Op tuỳ biến "FusedConcat": CONCATENATION int8 có kèm đổi scale/zero_point của từng đầu vào (thay cho
// QUANTIZE + CONCATENATION). Inputs: x_0..x_{k-1}, rồi k bảng 256 byte (lut[q+128]); bảng đồng nhất
// được nhận ra lúc Prepare và chỉ sao chép (memcpy). Trục nối suy ra từ shape.
// Dùng: resolver.AddCustom("FusedConcat", &reg) với reg = RegisterConcatLut().
TFLMRegistration RegisterConcatLut();
