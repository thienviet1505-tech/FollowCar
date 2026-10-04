#include "ConvOpt.h"

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

#include <common_functions.h>

// Chép filter theo từng khối kênh ra vào SRAM nội trước khi tính (xem ConvOpt.h).
// Tắt để so sánh: -DCONV_OPT_STAGE_FILTER=0.  Dung lượng vùng đệm: -DCONV_OPT_STAGE_BYTES=...
#ifndef CONV_OPT_STAGE_FILTER
#define CONV_OPT_STAGE_FILTER 1
#endif
#ifndef CONV_OPT_STAGE_BYTES
#define CONV_OPT_STAGE_BYTES 24576
#endif
#if CONV_OPT_STAGE_FILTER
#include <esp_heap_caps.h>
#endif

namespace {

// Khai báo extern "C" đặt TRONG namespace ẩn danh: common_functions.h đã khai báo cùng tên
// với linkage C++ ở phạm vi toàn cục (mangled, không có trong libesp-nn), nên khai báo ở
// phạm vi toàn cục sẽ bị báo "conflicting declaration". Trong namespace này, lời gọi
// bên dưới chọn đúng bản extern "C".
extern "C" int32_t esp_nn_dot_s8_aligned_esp32s3(
    const int8_t* input, const int8_t* filter, int length);

constexpr int kKernel = 3;

// Danh sách shape chạy per-pixel (đo thực tế: 7x7x256->64 per-pixel nhanh hơn đường staged).
const ConvOptTarget kDefaultTargets[] = {{7, 7, 256, 64}};
const ConvOptTarget* g_targets = kDefaultTargets;
int g_target_count = sizeof(kDefaultTargets) / sizeof(kDefaultTargets[0]);
int g_fast_nodes = 0;

#if CONV_OPT_STAGE_FILTER
// Vùng đệm SRAM nội dùng chung cho mọi node target (các node chạy tuần tự), chứa filter của
// một khối kênh ra. Cấp một lần, giữ suốt chương trình. Không cấp được thì node đọc filter
// trực tiếp từ PSRAM.
int8_t* g_stage = nullptr;
size_t g_stage_size = 0;

void EnsureStaging(size_t bytes) {
    if (bytes <= g_stage_size) return;
    int8_t* p = static_cast<int8_t*>(
        heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (p == nullptr) return;  // giữ vùng cũ (nếu có); node cần nhiều hơn sẽ đọc từ PSRAM
    if (g_stage != nullptr) heap_caps_free(g_stage);
    g_stage = p;
    g_stage_size = bytes;
}
#endif

struct ConvOptData {
    TFLMRegistration base;   // kernel CONV_2D gốc, dùng cho mọi node không phải target
    void* base_user_data;
    bool fast;               // node này chạy đường per-pixel
    int scratch_idx;
    int in_h, in_w, in_c;
    int out_c;
    int window_len;          // 3*3*in_c, luôn chia hết cho 16
    int32_t input_offset;    // = -zero_point của input
    int32_t output_offset;
    int32_t* mult;           // [out_c]
    int32_t* shift;          // [out_c]
    int32_t* correction;     // [out_c] = bias + tổng filter * input_offset
};

bool InTargetList(int in_h, int in_w, int in_c, int out_c) {
    for (int i = 0; i < g_target_count; ++i) {
        const ConvOptTarget& t = g_targets[i];
        if (t.in_h == in_h && t.in_w == in_w && t.in_c == in_c && t.out_c == out_c) return true;
    }
    return false;
}

// Shape/tham số có thuộc đường per-pixel không (chưa xét lượng tử hoá).
bool ShapeMatches(const TfLiteTensor* input, const TfLiteTensor* filter,
                  const TfLiteTensor* output, const TfLiteConvParams& p) {
    if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8) return false;
    if (input->dims->size != 4 || filter->dims->size != 4 || output->dims->size != 4) return false;

    const int in_h = input->dims->data[1];
    const int in_w = input->dims->data[2];
    const int in_c = input->dims->data[3];
    const int out_c = filter->dims->data[0];

    return input->dims->data[0] == 1 && output->dims->data[0] == 1 &&
           filter->dims->data[1] == kKernel && filter->dims->data[2] == kKernel &&
           filter->dims->data[3] == in_c &&
           output->dims->data[1] == in_h && output->dims->data[2] == in_w &&
           output->dims->data[3] == out_c &&
           (in_c % 16) == 0 &&
           p.stride_width == 1 && p.stride_height == 1 &&
           p.dilation_width_factor == 1 && p.dilation_height_factor == 1 &&
           p.padding == kTfLitePaddingSame &&
           p.activation == kTfLiteActNone &&
           InTargetList(in_h, in_w, in_c, out_c);
}

// Một cửa sổ 3x3xin_c quanh (out_y, out_x); ô ngoài biên điền pad_val (= zero_point input).
void BuildWindow(const ConvOptData* d, const int8_t* input, int8_t* window,
                 int out_y, int out_x, int8_t pad_val) {
    const int in_c = d->in_c;
    const int row_stride = d->in_w * in_c;

    for (int fy = 0; fy < kKernel; ++fy) {
        const int iy = out_y + fy - 1;
        int8_t* dst = window + fy * kKernel * in_c;

        if (iy < 0 || iy >= d->in_h) {
            memset(dst, pad_val, kKernel * in_c);
            continue;
        }

        const int x0 = out_x - 1;
        if (x0 >= 0 && x0 + kKernel <= d->in_w) {
            memcpy(dst, input + iy * row_stride + x0 * in_c, kKernel * in_c);
            continue;
        }

        for (int fx = 0; fx < kKernel; ++fx) {
            const int ix = x0 + fx;
            if (ix < 0 || ix >= d->in_w) {
                memset(dst + fx * in_c, pad_val, in_c);
            } else {
                memcpy(dst + fx * in_c, input + iy * row_stride + ix * in_c, in_c);
            }
        }
    }
}

// Duyệt theo khối kênh ra: với mỗi khối, chép filter của khối sang SRAM nội (nếu có vùng đệm)
// rồi chạy hết mọi điểm ảnh, nên mỗi filter chỉ đọc 1 lần từ PSRAM thay vì 1 lần mỗi điểm ảnh.
// Không có vùng đệm: 1 khối duy nhất, đọc filter thẳng từ PSRAM (cách cũ).
void RunFast(const ConvOptData* d, const int8_t* input, const int8_t* filter,
             int8_t* output, void* scratch) {
    int8_t* window = reinterpret_cast<int8_t*>(
        (reinterpret_cast<uintptr_t>(scratch) + 15u) & ~uintptr_t(15u));
    const int8_t pad_val = static_cast<int8_t>(-d->input_offset);

    int chunk_oc = d->out_c;
    int8_t* stage = nullptr;
#if CONV_OPT_STAGE_FILTER
    if (g_stage != nullptr) {
        const int fit = static_cast<int>(g_stage_size / d->window_len);
        if (fit >= 1) {
            stage = g_stage;
            chunk_oc = fit < d->out_c ? fit : d->out_c;
        }
    }
#endif

    for (int oc0 = 0; oc0 < d->out_c; oc0 += chunk_oc) {
        const int n = (d->out_c - oc0) < chunk_oc ? (d->out_c - oc0) : chunk_oc;
        const int8_t* w = filter + static_cast<size_t>(oc0) * d->window_len;
        if (stage != nullptr) {
            memcpy(stage, w, static_cast<size_t>(n) * d->window_len);
            w = stage;
        }

        for (int oy = 0; oy < d->in_h; ++oy) {
            for (int ox = 0; ox < d->in_w; ++ox) {
                BuildWindow(d, input, window, oy, ox, pad_val);

                int8_t* out = output + (oy * d->in_w + ox) * d->out_c + oc0;

                for (int i = 0; i < n; ++i) {
                    const int oc = oc0 + i;
                    int32_t acc = esp_nn_dot_s8_aligned_esp32s3(
                        window, w + static_cast<size_t>(i) * d->window_len, d->window_len);
                    acc += d->correction[oc];
                    acc = esp_nn_multiply_by_quantized_mult(acc, d->mult[oc], d->shift[oc]);
                    acc += d->output_offset;
                    acc = acc < -128 ? -128 : (acc > 127 ? 127 : acc);
                    out[i] = static_cast<int8_t>(acc);
                }
            }
        }
    }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* data = static_cast<ConvOptData*>(
        context->AllocatePersistentBuffer(context, sizeof(ConvOptData)));
    if (data == nullptr) return nullptr;

    memset(data, 0, sizeof(*data));
    data->base = tflite::Register_CONV_2D();
    data->base_user_data = data->base.init ? data->base.init(context, buffer, length) : nullptr;
    data->scratch_idx = -1;
    return data;
}

// Chạy Prepare của kernel gốc cho node không dùng đường per-pixel.
TfLiteStatus PrepareBase(TfLiteContext* context, TfLiteNode* node, ConvOptData* data) {
    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.prepare ? data->base.prepare(context, node) : kTfLiteOk;
    node->user_data = data;
    return status;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    if (node == nullptr || node->builtin_data == nullptr || node->user_data == nullptr) {
        return kTfLiteError;
    }

    auto* data = static_cast<ConvOptData*>(node->user_data);
    const auto& params = *static_cast<const TfLiteConvParams*>(node->builtin_data);
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* filter = micro_context->AllocateTempInputTensor(node, 1);
    if (input == nullptr || filter == nullptr || output == nullptr) return kTfLiteError;

    // Điều kiện đường per-pixel: shape trong danh sách, filter đối xứng (zero_point = 0),
    // lượng tử hoá theo kênh (hoặc 1 giá trị) và dữ liệu filter căn 16 byte (cần cho dot aligned).
    const TfLiteEvalTensor* filter_eval = tflite::micro::GetEvalInput(context, node, 1);
    const int8_t* w = filter_eval ? tflite::micro::GetTensorData<int8_t>(filter_eval) : nullptr;

    bool fast = w != nullptr && ((reinterpret_cast<uintptr_t>(w) & 15u) == 0) &&
                ShapeMatches(input, filter, output, params) &&
                filter->quantization.type == kTfLiteAffineQuantization &&
                filter->quantization.params != nullptr;

    const TfLiteAffineQuantization* affine = nullptr;
    int nscale = 0;
    if (fast) {
        affine = static_cast<const TfLiteAffineQuantization*>(filter->quantization.params);
        nscale = affine->scale->size;
        const int nzp = affine->zero_point->size;
        const int out_c = filter->dims->data[0];
        if ((nscale != 1 && nscale != out_c) || (nzp != 1 && nzp != out_c)) {
            fast = false;
        } else {
            for (int oc = 0; oc < out_c; ++oc) {
                if (affine->zero_point->data[nzp == 1 ? 0 : oc] != 0) { fast = false; break; }
            }
        }
    }
    data->fast = fast;

    if (!fast) {
        micro_context->DeallocateTempTfLiteTensor(output);
        micro_context->DeallocateTempTfLiteTensor(input);
        micro_context->DeallocateTempTfLiteTensor(filter);
        return PrepareBase(context, node, data);
    }

    data->in_h = input->dims->data[1];
    data->in_w = input->dims->data[2];
    data->in_c = input->dims->data[3];
    data->out_c = filter->dims->data[0];
    data->window_len = kKernel * kKernel * data->in_c;
    data->input_offset = -input->params.zero_point;
    data->output_offset = output->params.zero_point;

    const int out_c = data->out_c;
    int32_t* buf = static_cast<int32_t*>(
        context->AllocatePersistentBuffer(context, 3 * out_c * sizeof(int32_t)));
    if (buf == nullptr) return kTfLiteError;
    data->mult = buf;
    data->shift = buf + out_c;
    data->correction = buf + 2 * out_c;

    const int32_t* bias = nullptr;
    if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
        const TfLiteEvalTensor* bias_eval = tflite::micro::GetEvalInput(context, node, 2);
        bias = bias_eval ? tflite::micro::GetTensorData<int32_t>(bias_eval) : nullptr;
    }

    for (int oc = 0; oc < out_c; ++oc) {
        const double real_multiplier =
            static_cast<double>(input->params.scale) *
            static_cast<double>(affine->scale->data[nscale == 1 ? 0 : oc]) /
            static_cast<double>(output->params.scale);

        int32_t quantized_multiplier = 0;
        int shift = 0;
        tflite::QuantizeMultiplier(real_multiplier, &quantized_multiplier, &shift);
        data->mult[oc] = quantized_multiplier;
        data->shift[oc] = shift;

        int32_t filter_sum = 0;
        const int8_t* wo = w + oc * data->window_len;
        for (int i = 0; i < data->window_len; ++i) filter_sum += wo[i];

        data->correction[oc] = (bias != nullptr ? bias[oc] : 0) + filter_sum * data->input_offset;
    }

    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(filter);

#if CONV_OPT_STAGE_FILTER
    {
        // Vùng đệm đủ cho 1 khối kênh ra (tối đa CONV_OPT_STAGE_BYTES). Nếu 1 kênh ra đã
        // lớn hơn ngân sách thì bỏ qua staging cho node này.
        int chunk = CONV_OPT_STAGE_BYTES / data->window_len;
        if (chunk > out_c) chunk = out_c;
        if (chunk >= 1) EnsureStaging(static_cast<size_t>(chunk) * data->window_len);
    }
#endif

    ++g_fast_nodes;
    return context->RequestScratchBufferInArena(context, data->window_len + 15, &data->scratch_idx);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    if (node == nullptr || node->user_data == nullptr) return kTfLiteError;
    auto* data = static_cast<ConvOptData*>(node->user_data);

    if (!data->fast) {
        node->user_data = data->base_user_data;
        TfLiteStatus status = data->base.invoke(context, node);
        node->user_data = data;
        return status;
    }

    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

    void* scratch = context->GetScratchBuffer(context, data->scratch_idx);
    if (scratch == nullptr) return kTfLiteError;

    RunFast(data,
            tflite::micro::GetTensorData<int8_t>(input),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorData<int8_t>(output),
            scratch);
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterConvOpt() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

void ConvOptSetTargets(const ConvOptTarget* targets, int count) {
    g_targets = targets;
    g_target_count = count;
}

int ConvOptFastNodeCount() {
    return g_fast_nodes;
}

int ConvOptStagingBytes() {
#if CONV_OPT_STAGE_FILTER
    return static_cast<int>(g_stage_size);
#else
    return 0;
#endif
}