#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel PAD int8 cho đệm không gian (NHWC, đệm đối xứng/bất kỳ ở H và W): copy theo từng hàng
// bằng memcpy/memset, thay cho kernel tham chiếu của TFLM (chậm vì xử lý từng phần tử).
// Trường hợp khác (không phải int8, có giá trị đệm riêng, đệm ở batch/kênh...) chạy kernel gốc.
// Dùng: resolver.AddPad(RegisterPadFast());
TFLMRegistration RegisterPadFast();
