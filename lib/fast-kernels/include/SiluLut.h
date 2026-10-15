#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Op tuỳ biến "FusedSiLU": thay cặp LOGISTIC + MUL (SiLU = x * sigmoid(x)) bằng một lần tra bảng 256 phần tử.
// Input 0: x (int8), input 1: bảng 256 byte (lut[q + 128]) được tính sẵn lúc export (notebook), output: int8 cùng shape.
// Dùng: resolver.AddCustom("FusedSiLU", RegisterSiluLut());
TFLMRegistration RegisterSiluLut();
