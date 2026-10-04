#pragma once
#include "tensorflow/lite/micro/micro_common.h"

// Kernel CONV_2D int8 thay thế cho resolver.AddConv2D().
//
// Với các conv 3x3, stride 1, SAME, int8 mà shape nằm trong danh sách "target",
// chạy đường per-pixel (im2col từng điểm ảnh rồi esp_nn_dot_s8_aligned_esp32s3 cho
// từng kênh ra; hệ số requantize và hiệu chỉnh bias tính 1 lần ở Prepare).
// Mọi conv khác được giao nguyên cho kernel CONV_2D gốc của TFLM/ESP-NN.
//
// Dùng:  resolver.AddConv2D(RegisterConvOpt());
TFLMRegistration RegisterConvOpt();

// Một shape conv 3x3 s1 SAME chạy per-pixel: feature map vào in_h x in_w x in_c, ra out_c kênh.
// Điều kiện của đường per-pixel: in_c chia hết cho 16.
struct ConvOptTarget {
    int in_h, in_w, in_c, out_c;
};

// Tuỳ chọn: thay danh sách target mặc định ({7, 7, 256, 64}).
// Gọi TRƯỚC khi tạo interpreter. Mảng phải tồn tại suốt chương trình (static/const).
void ConvOptSetTargets(const ConvOptTarget* targets, int count);

// Số node CONV_2D đang chạy bằng đường per-pixel (sau AllocateTensors) - để kiểm tra
// danh sách target có khớp model hay không.
int ConvOptFastNodeCount();
