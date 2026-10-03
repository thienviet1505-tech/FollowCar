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
    bool invoke();

    // In profile của lần Invoke đầu tiên ra Serial.
    // Profile được tự động in sau lần invoke đầu tiên nên main.cpp không cần sửa.
    void printProfile();

    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    int inputChannels() const { return input_channels_; }
    size_t inputBytes() const { return input_bytes_; }
    float inputScale() const { return input_scale_; }
    int inputZeroPoint() const { return input_zero_point_; }

    const float* outputData() const { return output_f_.data(); }
    int outputElements() const;
    int outputNumDims() const;
    int outputDim(int index) const;
    float outputScale() const { return output_scale_; }
    int outputZeroPoint() const { return output_zero_point_; }

private:
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
