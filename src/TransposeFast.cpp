#include "TransposeFast.h"

#include <string.h>

#include "FastOpsCore.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

struct TransposeFastData {
    TFLMRegistration base;   // kernel TRANSPOSE gốc, dùng khi không vào đường nhanh
    void* base_user_data;
    bool fast;
    fastops::TransposePlan plan;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* data = static_cast<TransposeFastData*>(
        context->AllocatePersistentBuffer(context, sizeof(TransposeFastData)));
    if (data == nullptr) return nullptr;
    memset(data, 0, sizeof(*data));
    data->base = tflite::Register_TRANSPOSE();
    data->base_user_data = data->base.init ? data->base.init(context, buffer, length) : nullptr;
    return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<TransposeFastData*>(node->user_data);

    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.prepare ? data->base.prepare(context, node) : kTfLiteOk;
    node->user_data = data;
    if (status != kTfLiteOk) return status;

    data->fast = false;
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    const TfLiteEvalTensor* perm = tflite::micro::GetEvalInput(context, node, 1);

    if (input != nullptr && perm != nullptr && input->type == kTfLiteInt8 && perm->type == kTfLiteInt32) {
        const int rank = input->dims->size;
        const int32_t* p = tflite::micro::GetTensorData<int32_t>(perm);
        if (p != nullptr && perm->dims->data[0] == rank && rank >= 1 &&
            rank <= fastops::kMaxTransposeRank) {
            int dims[fastops::kMaxTransposeRank];
            int pm[fastops::kMaxTransposeRank];
            for (int i = 0; i < rank; ++i) {
                dims[i] = input->dims->data[i];
                pm[i] = p[i];
            }
            data->fast = fastops::MakeTransposePlan(dims, rank, pm, &data->plan);
        }
    }

    if (input != nullptr) micro_context->DeallocateTempTfLiteTensor(input);
    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    auto* data = static_cast<TransposeFastData*>(node->user_data);

    if (!data->fast) {
        node->user_data = data->base_user_data;
        TfLiteStatus status = data->base.invoke(context, node);
        node->user_data = data;
        return status;
    }

    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    fastops::TransposeS8(data->plan, tflite::micro::GetTensorData<int8_t>(input),
                         tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterTransposeFast() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
