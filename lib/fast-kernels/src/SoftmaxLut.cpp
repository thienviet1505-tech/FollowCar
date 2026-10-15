#include "SoftmaxLut.h"

#include <string.h>

#include "FastOpsCore.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

struct SoftmaxLutData {
    TFLMRegistration base;   // kernel SOFTMAX gốc, dùng khi không vào đường nhanh
    void* base_user_data;
    bool fast;
    float* table;            // [256]: exp(-beta * scale_in * d)
    int depth;               // kích thước trục cuối
    int rows;                // số hàng = tổng phần tử / depth
    float inv_scale_out;
    int zero_point_out;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* data = static_cast<SoftmaxLutData*>(
        context->AllocatePersistentBuffer(context, sizeof(SoftmaxLutData)));
    if (data == nullptr) return nullptr;
    memset(data, 0, sizeof(*data));
    data->base = tflite::Register_SOFTMAX();
    data->base_user_data = data->base.init ? data->base.init(context, buffer, length) : nullptr;
    return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<SoftmaxLutData*>(node->user_data);

    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.prepare ? data->base.prepare(context, node) : kTfLiteOk;
    node->user_data = data;
    if (status != kTfLiteOk) return status;

    data->fast = false;
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    const auto* params = static_cast<const TfLiteSoftmaxParams*>(node->builtin_data);

    if (input != nullptr && output != nullptr && params != nullptr &&
        input->type == kTfLiteInt8 && output->type == kTfLiteInt8 &&
        input->dims->size >= 1 && input->bytes == output->bytes) {
        const int depth = input->dims->data[input->dims->size - 1];
        float* table = static_cast<float*>(context->AllocatePersistentBuffer(context, 256 * sizeof(float)));
        if (depth > 0 && table != nullptr && output->params.scale > 0.0f) {
            fastops::BuildSoftmaxTable(table, params->beta, input->params.scale);
            data->table = table;
            data->depth = depth;
            data->rows = static_cast<int>(input->bytes) / depth;
            data->inv_scale_out = 1.0f / output->params.scale;
            data->zero_point_out = output->params.zero_point;
            data->fast = true;
        }
    }

    if (input != nullptr) micro_context->DeallocateTempTfLiteTensor(input);
    if (output != nullptr) micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<SoftmaxLutData*>(node->user_data);

    if (!data->fast) {
        node->user_data = data->base_user_data;
        TfLiteStatus status = data->base.invoke(context, node);
        node->user_data = data;
        return status;
    }

    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    fastops::SoftmaxRowsS8(tflite::micro::GetTensorData<int8_t>(input),
                           tflite::micro::GetTensorData<int8_t>(output),
                           data->rows, data->depth, data->table, data->inv_scale_out, data->zero_point_out);
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterSoftmaxLut() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
