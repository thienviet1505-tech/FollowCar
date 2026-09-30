#pragma once
#include <cstdint>
#include <cstddef>

// Wrapper cho TFLite Micro — chỉ hở ra kiểu dữ liệu cơ bản, để các module
// khác (Detector, main.cpp) không phải phụ thuộc trực tiếp vào TFLite.
class ModelInference {
public:
    // tensor_arena_size: vùng nhớ scratch cho TFLite Micro (cấp phát trong PSRAM).
    // Model cần khoảng 3.81MB activation, dùng 5MB để có dư địa.
    bool begin(size_t tensor_arena_size = 5 * 1024 * 1024);

    // input_data phải là float32 và đúng kích thước input model.
    bool invoke(const float* input_data, size_t input_len);

    int inputWidth() const    { return input_width_; }
    int inputHeight() const   { return input_height_; }
    int inputChannels() const { return input_channels_; }
    size_t inputBytes() const { return input_bytes_; }
    float inputScale() const  { return input_scale_; }
    int inputZeroPoint() const { return input_zero_point_; }

    const float* outputData() const;
    int outputElements() const;
    int outputNumDims() const;
    int outputDim(int index) const;
    float outputScale() const;
    int outputZeroPoint() const;

private:
    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;
    size_t input_bytes_ = 0;
    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
};