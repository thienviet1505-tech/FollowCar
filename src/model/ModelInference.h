#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

// Wrapper cho TFLite Micro — chỉ hở ra kiểu dữ liệu cơ bản, để các module
// khác (Detector, main.cpp) không phải phụ thuộc trực tiếp vào TFLite.
//
// Model dùng IO int8 (full integer quantization): input và output tensor đều là int8.
class ModelInference {
public:
    // Tensor arena được cấp phát trong PSRAM.
    bool begin(size_t tensor_arena_size = 1 * 1024 * 1024);

    // Ghi trực tiếp vào tensor input (inputBytes() byte int8, layout NCHW).
    int8_t* inputData();
    const int8_t* pixelLut() const { return pixel_lut_; }

    // Chạy model, sau đó giải lượng tử output sang float.
    // (Nếu bật PROFILE_OPS, profile của lần invoke đầu tiên được tự động in ra Serial.)
    bool invoke();

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    size_t inputBytes() const { return input_bytes_; }
    // true: input là [1,H,W,3] (NHWC); false: [1,3,H,W] (NCHW). Quyết định cách ghi ảnh vào inputData().
    bool inputIsNhwc() const { return input_nhwc_; }

    const float* outputData() const { return output_f_.data(); }
    int outputNumDims() const;
    int outputDim(int index) const;

private:
    bool input_nhwc_ = false;
    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;
    size_t input_bytes_ = 0;
    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;

    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;

    int8_t pixel_lut_[256] = {0};
    std::vector<float> output_f_;
};
