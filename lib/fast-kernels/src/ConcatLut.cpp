#include "ConcatLut.h"
#include <string.h>
#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

constexpr int kMaxIn = 8;

struct ConcatData {
    int k;
    int outer;
    int out_row;
    int in_row[kMaxIn];
    int col[kMaxIn];       // vị trí bắt đầu của đầu vào i trong một hàng đầu ra
    bool ident[kMaxIn];
};

void* Init(TfLiteContext* context, const char*, size_t) {
    return context->AllocatePersistentBuffer(context, sizeof(ConcatData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    auto* d = static_cast<ConcatData*>(node->user_data);
    const int total = node->inputs->size;
    TF_LITE_ENSURE(context, total >= 2 && (total % 2) == 0);
    const int k = total / 2;
    TF_LITE_ENSURE(context, k <= kMaxIn);
    d->k = k;

    tflite::MicroContext* mc = tflite::GetMicroContext(context);
    TfLiteTensor* out = mc->AllocateTempOutputTensor(node, 0);
    TF_LITE_ENSURE(context, out != nullptr);
    TF_LITE_ENSURE_EQ(context, out->type, kTfLiteInt8);

    // Trục nối: chiều duy nhất mà đầu ra khác đầu vào đầu tiên (k >= 2 nên luôn xác định).
    TfLiteTensor* in0 = mc->AllocateTempInputTensor(node, 0);
    TF_LITE_ENSURE(context, in0 != nullptr);
    TF_LITE_ENSURE_EQ(context, in0->dims->size, out->dims->size);
    int axis = -1;
    for (int i = 0; i < out->dims->size; ++i) {
        if (out->dims->data[i] != in0->dims->data[i]) {
            TF_LITE_ENSURE_EQ(context, axis, -1);
            axis = i;
        }
    }
    TF_LITE_ENSURE(context, axis >= 0);
    mc->DeallocateTempTfLiteTensor(in0);

    int outer = 1, out_row = 1;
    for (int i = 0; i < axis; ++i) outer *= out->dims->data[i];
    for (int i = axis; i < out->dims->size; ++i) out_row *= out->dims->data[i];
    d->outer = outer;
    d->out_row = out_row;

    int col = 0;
    int inner_after = out_row / out->dims->data[axis];  // tích các chiều sau trục
    for (int i = 0; i < k; ++i) {
        TfLiteTensor* in = mc->AllocateTempInputTensor(node, i);
        TfLiteTensor* tab = mc->AllocateTempInputTensor(node, k + i);
        TF_LITE_ENSURE(context, in != nullptr && tab != nullptr);
        TF_LITE_ENSURE_EQ(context, in->type, kTfLiteInt8);
        TF_LITE_ENSURE_EQ(context, tab->type, kTfLiteInt8);
        TF_LITE_ENSURE_EQ(context, tab->dims->data[0], 256);
        d->in_row[i] = in->dims->data[axis] * inner_after;
        d->col[i] = col;
        col += d->in_row[i];
        bool ident = true;
        for (int q = -128; q <= 127; ++q) {
            if (tab->data.int8[q + 128] != q) {
                ident = false;
                break;
            }
        }
        d->ident[i] = ident;
        mc->DeallocateTempTfLiteTensor(in);
        mc->DeallocateTempTfLiteTensor(tab);
    }
    TF_LITE_ENSURE_EQ(context, col, out_row);
    mc->DeallocateTempTfLiteTensor(out);
    return kTfLiteOk;
}

inline void LutRow(const uint8_t* lut, const int8_t* src, int8_t* dst, int n) {
    int j = 0;
    if (((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 3u) == 0) {
        const uint32_t* s32 = reinterpret_cast<const uint32_t*>(src);
        uint32_t* d32 = reinterpret_cast<uint32_t*>(dst);
        const int n4 = n >> 2;
        for (int w = 0; w < n4; ++w) {
            const uint32_t v = s32[w] ^ 0x80808080u;
            d32[w] = static_cast<uint32_t>(lut[v & 0xFFu]) | (static_cast<uint32_t>(lut[(v >> 8) & 0xFFu]) << 8) |
                     (static_cast<uint32_t>(lut[(v >> 16) & 0xFFu]) << 16) | (static_cast<uint32_t>(lut[v >> 24]) << 24);
        }
        j = n4 << 2;
    }
    for (; j < n; ++j) dst[j] = static_cast<int8_t>(lut[static_cast<uint8_t>(src[j]) ^ 0x80u]);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    const auto* d = static_cast<const ConcatData*>(node->user_data);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);

    alignas(16) uint8_t lut[256];
    for (int i = 0; i < d->k; ++i) {
        const TfLiteEvalTensor* in = tflite::micro::GetEvalInput(context, node, i);
        const int8_t* src = tflite::micro::GetTensorData<int8_t>(in);
        int8_t* dst = out + d->col[i];
        const int n = d->in_row[i];
        if (d->ident[i]) {
            for (int o = 0; o < d->outer; ++o) memcpy(dst + static_cast<size_t>(o) * d->out_row, src + static_cast<size_t>(o) * n, n);
        } else {
            const TfLiteEvalTensor* tab = tflite::micro::GetEvalInput(context, node, d->k + i);
            memcpy(lut, tflite::micro::GetTensorData<int8_t>(tab), 256);
            for (int o = 0; o < d->outer; ++o) LutRow(lut, src + static_cast<size_t>(o) * n, dst + static_cast<size_t>(o) * d->out_row, n);
        }
    }
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterConcatLut() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}
