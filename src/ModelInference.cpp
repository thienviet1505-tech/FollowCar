#include "ModelInference.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <esp_heap_caps.h>
#include "LogisticLut.h"
#include "QuantizeLut.h"
#include "ConvOpt.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// Tắt từng phần để so sánh: thêm vào build_flags  -DMODEL_IN_PSRAM=0  hoặc  -DNN_DUAL_CORE=0
#ifndef MODEL_IN_PSRAM
#define MODEL_IN_PSRAM 1
#endif
#ifndef NN_DUAL_CORE
#define NN_DUAL_CORE 1
#endif
// Profiler chỉ bật khi cần:  -DPROFILE_OPS=1  (in bảng thời gian từng op sau mỗi lần invoke).
#ifndef PROFILE_OPS
#define PROFILE_OPS 1
#endif

// Đa lõi của ESP-NN (esp-nn/include/esp_nn_multicore.h).
extern "C" {
void esp_nn_dual_core_enable(void);
bool esp_nn_dual_core_active(void);
void* esp_nn_dual_core_scratch(int size);
}

// Cấp sẵn scratch dùng chung của ESP-NN dual-core ngay khi khởi động (byte).
// Lớp conv 3x3 K=1152 cần 62.560 B; nếu để thư viện tự tăng dần, lúc cần thì RAM nội
// đã phân mảnh (khối lớn nhất < 62,5 KB) -> rơi sang PSRAM -> các lớp K=1152 chậm.
// Đặt 0 để tắt.
#ifndef NN_SCRATCH_PREALLOC
#define NN_SCRATCH_PREALLOC 65536
#endif

namespace {
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
uint8_t* tensor_arena = nullptr;

#if PROFILE_OPS
// Profiler phải sống ít nhất lâu bằng interpreter.
static tflite::MicroProfiler profiler;
static bool profile_printed = false;
#define PROFILER_PTR (&profiler)
#else
#define PROFILER_PTR nullptr
#endif

constexpr int kNumOps = 103;

tflite::MicroMutableOpResolver<kNumOps>& GetResolver() {
    static tflite::MicroMutableOpResolver<kNumOps> resolver;
    static bool initialized = false;

    if (!initialized) {
        resolver.AddAbs();
        resolver.AddAdd();
        resolver.AddAddN();
        resolver.AddArgMax();
        resolver.AddArgMin();
        resolver.AddAssignVariable();
        resolver.AddAveragePool2D();
        resolver.AddBatchMatMul();
        resolver.AddBatchToSpaceNd();
        resolver.AddBroadcastArgs();
        resolver.AddBroadcastTo();
        resolver.AddCallOnce();
        resolver.AddCast();
        resolver.AddCeil();
        resolver.AddCircularBuffer();
        resolver.AddConcatenation();
        resolver.AddConv2D(RegisterConvOpt());
        resolver.AddCos();
        resolver.AddCumSum();
        resolver.AddDecode();
        resolver.AddDepthToSpace();
        resolver.AddDepthwiseConv2D();
        resolver.AddDequantize();
        resolver.AddDetectionPostprocess();
        resolver.AddDiv();
        resolver.AddDynamicUpdateSlice();
        resolver.AddElu();
        resolver.AddEmbeddingLookup();
        resolver.AddEqual();
        resolver.AddExp();
        resolver.AddExpandDims();
        resolver.AddFill();
        resolver.AddFloor();
        resolver.AddFloorDiv();
        resolver.AddFloorMod();
        resolver.AddFullyConnected();
        resolver.AddGather();
        resolver.AddGatherNd();
        resolver.AddGreater();
        resolver.AddGreaterEqual();
        resolver.AddHardSwish();
        resolver.AddIf();
        resolver.AddL2Normalization();
        resolver.AddL2Pool2D();
        resolver.AddLeakyRelu();
        resolver.AddLess();
        resolver.AddLessEqual();
        resolver.AddLog();
        resolver.AddLogSoftmax();
        resolver.AddLogicalAnd();
        resolver.AddLogicalNot();
        resolver.AddLogicalOr();
        resolver.AddLogistic(RegisterLogisticLut());
        resolver.AddMaxPool2D();
        resolver.AddMaximum();
        resolver.AddMean();
        resolver.AddMinimum();
        resolver.AddMirrorPad();
        resolver.AddMul();
        resolver.AddNeg();
        resolver.AddNotEqual();
        resolver.AddPack();
        resolver.AddPad();
        resolver.AddPadV2();
        resolver.AddPrelu();
        resolver.AddQuantize(RegisterQuantizeLut());
        resolver.AddReadVariable();
        resolver.AddReduceAll();
        resolver.AddReduceMax();
        resolver.AddReduceMin();
        resolver.AddRelu();
        resolver.AddRelu6();
        resolver.AddReshape();
        resolver.AddResizeBilinear();
        resolver.AddResizeNearestNeighbor();
        resolver.AddReverseV2();
        resolver.AddRound();
        resolver.AddRsqrt();
        resolver.AddSelectV2();
        resolver.AddShape();
        resolver.AddSin();
        resolver.AddSlice();
        resolver.AddSoftmax();
        resolver.AddSpaceToBatchNd();
        resolver.AddSpaceToDepth();
        resolver.AddSplit();
        resolver.AddSplitV();
        resolver.AddSqrt();
        resolver.AddSquare();
        resolver.AddSquaredDifference();
        resolver.AddSqueeze();
        resolver.AddStridedSlice();
        resolver.AddSub();
        resolver.AddSum();
        resolver.AddSvdf();
        resolver.AddTanh();
        resolver.AddTranspose();
        resolver.AddTransposeConv();
        resolver.AddUnidirectionalSequenceLSTM();
        resolver.AddUnpack();
        resolver.AddVarHandle();
        resolver.AddWhile();
        resolver.AddZerosLike();
        initialized = true;
    }

    return resolver;
}

// Chép model từ flash sang PSRAM (căn 16 byte như model_data) để đọc trọng số nhanh hơn.
const unsigned char* modelInPsram() {
    unsigned char* p = (unsigned char*)heap_caps_aligned_alloc(16, model_data_len, MALLOC_CAP_SPIRAM);
    if (!p) {
        Serial.println("[Model] Khong du PSRAM de chep model, dung ban trong flash");
        return model_data;
    }
    const uint32_t t0 = millis();
    memcpy(p, model_data, model_data_len);
    Serial.printf("[Model] Da chep model vao PSRAM: %u byte, %lu ms\n",
                  (unsigned)model_data_len, (unsigned long)(millis() - t0));
    return p;
}
}  // namespace

bool ModelInference::begin(size_t tensor_arena_size) {
#if MODEL_IN_PSRAM
    tfl_model = tflite::GetModel(modelInPsram());
#else
    tfl_model = tflite::GetModel(model_data);
#endif
    if (!tfl_model) {
        Serial.println("[ModelInference] tflite::GetModel tra ve null, kiem tra model_data.h");
        return false;
    }

    tensor_arena = static_cast<uint8_t*>(
        heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
    );

    if (!tensor_arena) {
        Serial.printf("[Model] ERROR: PSRAM arena allocation failed (%u bytes)\n",
                      (unsigned)tensor_arena_size);
        return false;
    }

    Serial.printf("[Model] PSRAM arena=%u bytes, free PSRAM=%u bytes\n",
                  (unsigned)tensor_arena_size,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    static tflite::MicroInterpreter static_interpreter(
        tfl_model, GetResolver(), tensor_arena, tensor_arena_size, nullptr, PROFILER_PTR);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ModelInference] AllocateTensors that bai");
        return false;
    }
    Serial.printf("[Model] ConvOpt per-pixel nodes: %d\n", ConvOptFastNodeCount());
    Serial.printf("[Model] ConvOpt staging=%d bytes | internal free=%u largest=%u\n",
                  ConvOptStagingBytes(),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

#if NN_DUAL_CORE
    esp_nn_dual_core_enable();   // gọi từ core chạy suy luận (loopTask)
    Serial.printf("[Model] ESP-NN dual-core: %s\n", esp_nn_dual_core_active() ? "ON" : "OFF");
#if NN_SCRATCH_PREALLOC > 0
    if (esp_nn_dual_core_active()) {
        void* scr = esp_nn_dual_core_scratch(NN_SCRATCH_PREALLOC);
        Serial.printf("[Model] dual-core scratch=%d B @%p (%s) | internal free=%u largest=%u\n",
                      NN_SCRATCH_PREALLOC, scr,
                      scr == nullptr ? "null"
                      : ((uintptr_t)scr >= 0x3C000000u && (uintptr_t)scr < 0x3E000000u) ? "PSRAM"
                      : "internal",
                      (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                      (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    }
#endif
#endif

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Chi ho tro model IO int8.
    if (input_tensor->type != kTfLiteInt8 || output_tensor->type != kTfLiteInt8) {
        Serial.printf("[ModelInference] Can model IO int8 (input type=%d, output type=%d, int8=%d)\n",
                      (int)input_tensor->type, (int)output_tensor->type, (int)kTfLiteInt8);
        return false;
    }

    // Model input shape: [1, C, H, W] = NCHW.
    input_channels_ = input_tensor->dims->data[1];
    input_height_ = input_tensor->dims->data[2];
    input_width_ = input_tensor->dims->data[3];

    input_bytes_ = input_tensor->bytes;
    input_scale_ = input_tensor->params.scale;
    input_zero_point_ = input_tensor->params.zero_point;

    output_scale_ = output_tensor->params.scale;
    output_zero_point_ = output_tensor->params.zero_point;
    output_f_.assign(output_tensor->bytes, 0.0f);

    // LUT: pixel 0..255 -> int8 quantized value.
    for (int v = 0; v < 256; v++) {
        int q = (int)std::lround((v / 255.0f) / input_scale_) + input_zero_point_;
        pixel_lut_[v] = (int8_t)std::max(-128, std::min(127, q));
    }

    Serial.printf("[Model] input=%dx%dx%d bytes=%d scale=%.6f zp=%d | output type=%d bytes=%d scale=%.6f zp=%d dims=",
                  input_width_, input_height_, input_channels_,
                  (int)input_bytes_, input_scale_, input_zero_point_,
                  output_tensor->type, (int)output_tensor->bytes,
                  output_scale_, output_zero_point_);
    for (int i = 0; i < output_tensor->dims->size; i++) {
        Serial.printf("%d", output_tensor->dims->data[i]);
        if (i + 1 < output_tensor->dims->size) Serial.print("x");
    }
    Serial.println();

    Serial.printf("[Model] arena used=%u / %u bytes\n",
                  (unsigned)interpreter->arena_used_bytes(), (unsigned)tensor_arena_size);

    return true;
}

int8_t* ModelInference::inputData() {
    return input_tensor ? input_tensor->data.int8 : nullptr;
}

bool ModelInference::invoke() {
    if (!interpreter || !input_tensor || !output_tensor) return false;

    Serial.println("[NN] INT8 invoke start");
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[NN] INT8 invoke FAIL");
        return false;
    }
    Serial.println("[NN] INT8 invoke end");

#if PROFILE_OPS
    // MicroProfiler chỉ chứa tối đa 4096 sự kiện (~283/lần invoke) -> in xong phải xoá,
    // nếu không sẽ abort() sau ~14 frame. Chỉ in 1 lần đầu cho đỡ rối log.
    if (!profile_printed) {
        profile_printed = true;
        Serial.println();
        Serial.println("================ TFLITE MICRO PROFILE ================");
        profiler.Log();
        Serial.println("=======================================================");
        Serial.println();
    }
    profiler.ClearEvents();
#endif

    // Giai luong tuong output: x = (q - zero_point) * scale.
    const int8_t* q = output_tensor->data.int8;
    const size_t n = output_f_.size();
    float* out = output_f_.data();
    for (size_t i = 0; i < n; i++) {
        out[i] = (float)((int)q[i] - output_zero_point_) * output_scale_;
    }
    return true;
}

void ModelInference::printProfile() {
#if PROFILE_OPS
    Serial.println();
    Serial.println("================ TFLITE MICRO PROFILE ================");
    profiler.Log();
    Serial.println("=======================================================");
    profiler.ClearEvents();
#else
    Serial.println("[Profile] Build voi -DPROFILE_OPS=1 de bat profiler.");
#endif
}

int ModelInference::outputElements() const {
    return output_tensor ? output_tensor->bytes : 0;
}

int ModelInference::outputNumDims() const {
    return output_tensor ? output_tensor->dims->size : 0;
}

int ModelInference::outputDim(int index) const {
    if (!output_tensor || index < 0 || index >= output_tensor->dims->size) return 0;
    return output_tensor->dims->data[index];
}