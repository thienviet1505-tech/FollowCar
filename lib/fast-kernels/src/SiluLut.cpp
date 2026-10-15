#include "SiluLut.h"
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

TfLiteStatus SiluPrepare(TfLiteContext* context, TfLiteNode* node) {
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
    TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* table = micro_context->AllocateTempInputTensor(node, 1);
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    TF_LITE_ENSURE(context, input != nullptr && table != nullptr && output != nullptr);
    TF_LITE_ENSURE_EQ(context, input->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, table->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, output->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, table->dims->size, 1);
    TF_LITE_ENSURE_EQ(context, table->dims->data[0], 256);

    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(table);
    micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus SiluEval(TfLiteContext* context, TfLiteNode* node) {
    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor* table = tflite::micro::GetEvalInput(context, node, 1);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

    const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
    const int n = tflite::micro::GetTensorShape(input).FlatSize();

    // Chép bảng về stack (SRAM nội, 256 byte) để tra nhanh, không phụ thuộc cache của flash/PSRAM.
    alignas(16) uint8_t lut[256];
    memcpy(lut, tflite::micro::GetTensorData<int8_t>(table), 256);

    // Chỉ số = (q + 128) = q ^ 0x80 ở dạng byte -> xor cả từ 32 bit với 0x80808080.
    int i = 0;
    if ((reinterpret_cast<uintptr_t>(in) & 3u) == 0 && (reinterpret_cast<uintptr_t>(out) & 3u) == 0) {
        const uint32_t* in32 = reinterpret_cast<const uint32_t*>(in);
        uint32_t* out32 = reinterpret_cast<uint32_t*>(out);
        const int n4 = n >> 2;
        for (int k = 0; k < n4; ++k) {
            const uint32_t w = in32[k] ^ 0x80808080u;
            out32[k] = static_cast<uint32_t>(lut[w & 0xFFu]) |
                       (static_cast<uint32_t>(lut[(w >> 8) & 0xFFu]) << 8) |
                       (static_cast<uint32_t>(lut[(w >> 16) & 0xFFu]) << 16) |
                       (static_cast<uint32_t>(lut[w >> 24]) << 24);
        }
        i = n4 << 2;
    }
    for (; i < n; ++i) out[i] = static_cast<int8_t>(lut[static_cast<uint8_t>(in[i]) ^ 0x80u]);
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterSiluLut() {
    return tflite::micro::RegisterOp(nullptr, SiluPrepare, SiluEval);
}
