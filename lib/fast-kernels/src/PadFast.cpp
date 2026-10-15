#include "PadFast.h"

#include <string.h>

#include "FastOpsCore.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

struct PadFastData {
    TFLMRegistration base;   // kernel PAD gốc, dùng khi không vào đường nhanh
    void* base_user_data;
    bool fast;
    int n, h, w, c;          // shape input
    int top, bottom, left, right;
    int8_t pad_value;        // = zero_point của output (cùng quy ước kernel gốc)
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* data = static_cast<PadFastData*>(context->AllocatePersistentBuffer(context, sizeof(PadFastData)));
    if (data == nullptr) return nullptr;
    memset(data, 0, sizeof(*data));
    data->base = tflite::Register_PAD();
    data->base_user_data = data->base.init ? data->base.init(context, buffer, length) : nullptr;
    return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<PadFastData*>(node->user_data);

    // Luôn chạy Prepare của kernel gốc để đường dự phòng dùng được.
    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.prepare ? data->base.prepare(context, node) : kTfLiteOk;
    node->user_data = data;
    if (status != kTfLiteOk) return status;

    data->fast = false;
    if (node->inputs->size != 2) return kTfLiteOk;   // có giá trị đệm riêng: để kernel gốc

    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    const TfLiteEvalTensor* paddings = tflite::micro::GetEvalInput(context, node, 1);

    if (input != nullptr && output != nullptr && paddings != nullptr &&
        input->type == kTfLiteInt8 && output->type == kTfLiteInt8 && paddings->type == kTfLiteInt32 &&
        input->dims->size == 4 && output->dims->size == 4) {
        const int32_t* p = tflite::micro::GetTensorData<int32_t>(paddings);   // [4][2]: trước/sau của từng chiều
        if (p != nullptr && p[0] == 0 && p[1] == 0 && p[6] == 0 && p[7] == 0 &&
            p[2] >= 0 && p[3] >= 0 && p[4] >= 0 && p[5] >= 0) {
            const int* in_d = input->dims->data;
            const int* out_d = output->dims->data;
            if (out_d[0] == in_d[0] && out_d[1] == in_d[1] + p[2] + p[3] &&
                out_d[2] == in_d[2] + p[4] + p[5] && out_d[3] == in_d[3]) {
                data->n = in_d[0];
                data->h = in_d[1];
                data->w = in_d[2];
                data->c = in_d[3];
                data->top = p[2];
                data->bottom = p[3];
                data->left = p[4];
                data->right = p[5];
                data->pad_value = static_cast<int8_t>(output->params.zero_point);
                data->fast = true;
            }
        }
    }

    if (input != nullptr) micro_context->DeallocateTempTfLiteTensor(input);
    if (output != nullptr) micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<PadFastData*>(node->user_data);

    if (!data->fast) {
        node->user_data = data->base_user_data;
        TfLiteStatus status = data->base.invoke(context, node);
        node->user_data = data;
        return status;
    }

    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    fastops::PadSpatialS8(tflite::micro::GetTensorData<int8_t>(input),
                          tflite::micro::GetTensorData<int8_t>(output),
                          data->n, data->h, data->w, data->c,
                          data->top, data->bottom, data->left, data->right, data->pad_value);
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterPadFast() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
