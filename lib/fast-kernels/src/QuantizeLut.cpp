#include "QuantizeLut.h"
#include <cmath>
#include <algorithm>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

struct LutData {
    int8_t lut[256];  // lut[q + 128] = giá trị int8 sau khi đổi scale/zero_point
};

void* LutInit(TfLiteContext* context, const char* /*buffer*/, size_t /*length*/) {
    return context->AllocatePersistentBuffer(context, sizeof(LutData));
}

TfLiteStatus LutPrepare(TfLiteContext* context, TfLiteNode* node) {
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    TF_LITE_ENSURE(context, input != nullptr);
    TF_LITE_ENSURE(context, output != nullptr);
    // Model hiện tại chỉ có QUANTIZE int8->int8; kiểu khác thì báo lỗi rõ ràng.
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);

    const float s_in = input->params.scale;
    const int zp_in = input->params.zero_point;
    const float s_out = output->params.scale;
    const int zp_out = output->params.zero_point;

    LutData* data = static_cast<LutData*>(node->user_data);
    for (int q = -128; q <= 127; q++) {
        const float x = (float)(q - zp_in) * s_in;
        const int r = (int)std::lround(x / s_out) + zp_out;
        data->lut[q + 128] = (int8_t)std::max(-128, std::min(127, r));
    }

    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus LutEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    const LutData* data = static_cast<const LutData*>(node->user_data);

    const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
    const int n = tflite::micro::GetTensorShape(input).FlatSize();
    const int8_t* lut = data->lut + 128;

    for (int i = 0; i < n; i++) out[i] = lut[in[i]];
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterQuantizeLut() {
    return tflite::micro::RegisterOp(LutInit, LutPrepare, LutEval);
}
