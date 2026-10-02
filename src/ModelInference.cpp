#include "ModelInference.h"
#include <Arduino.h>
#include <cstring>
#include <esp_heap_caps.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

namespace {
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
uint8_t* tensor_arena = nullptr;

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
        resolver.AddConv2D();
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
        resolver.AddLogistic();
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
        resolver.AddQuantize();
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
}  // namespace

bool ModelInference::begin(size_t tensor_arena_size) {
    tfl_model = tflite::GetModel(model_data);
    if (!tfl_model) {
        Serial.println("[ModelInference] tflite::GetModel tra ve null, kiem tra model_data.h");
        return false;
    }

    tensor_arena = (uint8_t*)heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM);
    if (!tensor_arena) {
        Serial.println("[ModelInference] Khong cap phat duoc tensor arena tu PSRAM");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        tfl_model, GetResolver(), tensor_arena, tensor_arena_size);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ModelInference] AllocateTensors that bai");
        return false;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Model input shape: [1, C, H, W] = NCHW
    input_channels_ = input_tensor->dims->data[1];
    input_height_   = input_tensor->dims->data[2];
    input_width_    = input_tensor->dims->data[3];

    input_bytes_      = input_tensor->bytes;
    input_scale_      = input_tensor->params.scale;
    input_zero_point_ = input_tensor->params.zero_point;

    Serial.printf("[Model] input=%dx%dx%d bytes=%d | output type=%d bytes=%d dims=",
                  input_width_, input_height_, input_channels_,
                  (int)input_bytes_, output_tensor->type, (int)output_tensor->bytes);
    for (int i = 0; i < output_tensor->dims->size; i++) {
        Serial.printf("%d", output_tensor->dims->data[i]);
        if (i + 1 < output_tensor->dims->size) Serial.print("x");
    }
    Serial.println();

    return true;
}

bool ModelInference::invoke(const float* input_data, size_t input_len) {
    if (!interpreter || !input_tensor) return false;

    if (input_len != input_bytes_) {
        Serial.printf("[ModelInference] Input size sai: %d != %d\n",
                      input_len, input_bytes_);
        return false;
    }

    memcpy(input_tensor->data.f, input_data, input_len);

    TfLiteStatus status = interpreter->Invoke();
    return status == kTfLiteOk;
}

const float* ModelInference::outputData() const {
    return output_tensor->data.f;
}

int ModelInference::outputElements() const {
    return output_tensor->bytes;
}

int ModelInference::outputNumDims() const {
    return output_tensor->dims->size;
}

int ModelInference::outputDim(int index) const {
    return output_tensor->dims->data[index];
}

float ModelInference::outputScale() const {
    return output_tensor->params.scale;
}

int ModelInference::outputZeroPoint() const {
    return output_tensor->params.zero_point;
}