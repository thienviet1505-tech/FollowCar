#include "ConvOpt.h"
#include "ConvOptTune.h"
#include "ConvCore.h"

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <common_functions.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>

// Dung lượng vùng đệm SRAM chứa filter của một khối kênh ra (mỗi lõi một vùng). Mặc định 24 KB.
#ifndef CONV_OPT_STAGE_BYTES
#define CONV_OPT_STAGE_BYTES 24576
#endif
// 1: cho phép chạy hai lõi (cần esp_nn_dual_core_enable() đã được gọi). 0: chỉ một lõi.
#ifndef CONV_OPT_DUAL
#define CONV_OPT_DUAL 1
#endif
// 1: in một dòng [Tune] cho mỗi conv sau khi đo.
#ifndef CONV_OPT_REPORT
#define CONV_OPT_REPORT 1
#endif

namespace {

// Khai báo extern "C" đặt TRONG namespace ẩn danh: common_functions.h đã khai báo cùng tên
// với linkage C++ ở phạm vi toàn cục (mangled, không có trong libesp-nn), nên khai báo ở
// phạm vi toàn cục sẽ bị báo "conflicting declaration". Trong namespace này, lời gọi
// bên dưới chọn đúng bản extern "C".
extern "C" int32_t esp_nn_dot_s8_aligned_esp32s3(
    const int8_t* input, const int8_t* filter, int length);
extern "C" bool esp_nn_dual_core_active(void);
extern "C" bool esp_nn_dual_core_run(void (*fn)(void*), void* arg);
extern "C" void esp_nn_dual_core_wait(void);
extern "C" void* esp_nn_dual_core_scratch(int size);
// Kernel 1x1 SIMD của ESP-NN (hợp ngữ, xử lý 8 điểm ảnh một lúc). Filter [out_ch][in_ch], in_ch bội 8,
// filter căn 16 byte; in_ch/out_ch/bias tính như conv 1x1 thường. scratch >= 24*in_ch + 8*out_ch + 64 byte.
extern "C" void esp_nn_conv_s8_mult8_1x1_esp32s3(
    const int8_t* input_data, const uint16_t input_wd, const uint16_t input_ht,
    const uint16_t in_channels, const int32_t input_offset, const int8_t* filter_aligned,
    const int32_t* bias, int8_t* out_data, const uint16_t out_wd, const uint16_t out_ht,
    const uint16_t out_channels, const int32_t out_offset, const int32_t* out_shift,
    const int32_t* out_mult, const int32_t activation_min, const int32_t activation_max,
    void* buffer);

struct DotS8 {
    int32_t operator()(const int8_t* a, const int8_t* b, int len) const {
        return esp_nn_dot_s8_aligned_esp32s3(a, b, len);
    }
};
struct ReqS8 {
    int32_t operator()(int32_t acc, int32_t mult, int32_t shift) const {
        return esp_nn_multiply_by_quantized_mult(acc, mult, shift);
    }
};

// Cách chạy một conv.
enum Mode { kBase = 0, kOne = 1, kDualOC = 2, kDualPx = 3, kIm2col = 4, kModeCount = 5 };
const char* const kModeName[kModeCount] = {"base", "one", "dual-oc", "dual-px", "im2col"};
constexpr int kNA = -1;    // cách này không dùng được (không có hai lõi, hoặc không chia được)
constexpr int kBad = -2;   // kết quả không khớp từng bit với kernel gốc

constexpr int kWindowBytes = 2048;   // cửa sổ 3x3xCin của mỗi lõi (wpad tối đa)
constexpr int kStageBytes = CONV_OPT_STAGE_BYTES;
constexpr int kCoreBytes = kWindowBytes + kStageBytes;

// Vùng SRAM nội của lõi chính: [cửa sổ | vùng đệm filter]. Cấp một lần, giữ suốt chương trình.
// Lõi phụ lấy vùng tương tự từ esp_nn_dual_core_scratch().
int8_t* g_core_buf = nullptr;

bool EnsureCoreBuffer() {
    if (g_core_buf != nullptr) return true;
    g_core_buf = static_cast<int8_t*>(
        heap_caps_aligned_alloc(16, kCoreBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    return g_core_buf != nullptr;
}

bool g_tune = false;
int g_node_seq = 0;
int g_eligible = 0;
int g_mode_count[kModeCount] = {0, 0, 0, 0, 0};
int64_t g_tune_us = 0;

struct ConvOptData {
    TFLMRegistration base;   // kernel CONV_2D gốc
    void* base_user_data;
    bool eligible;           // conv 3x3 đủ điều kiện chạy bằng kernel ConvOpt
    bool tuned;
    int mode;
    int index;               // thứ tự conv trong model (khớp bảng profile)
    convopt::Geom g;
    convopt::Quant q;
    int8_t* packed;          // filter đã căn (wpad byte mỗi hàng) nằm ở SRAM nội; nullptr nếu dùng filter gốc
    bool i2c_ok;             // mode im2col dùng được (số điểm ảnh chia hết 8, vùng SRAM đủ chứa một khúc)
    int i2c_chunk;           // số điểm ảnh mỗi khúc im2col (bội 8)
    const int32_t* bias_raw; // [out_c] bias gốc (đường 1x1 tự cộng phần zero_point)
    int32_t input_offset;    // = -zero_point của input
};

TfLiteStatus BaseInvoke(TfLiteContext* context, TfLiteNode* node, ConvOptData* data) {
    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.invoke(context, node);
    node->user_data = data;
    return status;
}

// ---------------------------------------------------------------- chạy kernel ConvOpt
void RunOne(const ConvOptData* d, const int8_t* in, const int8_t* filt, int8_t* out,
            int8_t* window, int8_t* stage, int p0, int p1, int oc0, int oc1) {
    convopt::RunTile(d->g, d->q, in, filt, out, window, stage, kStageBytes, p0, p1, oc0, oc1,
                     DotS8(), ReqS8());
}

struct Job {
    const ConvOptData* d;
    const int8_t* in;
    const int8_t* filt;
    int8_t* out;
    int8_t* window;
    int8_t* stage;
    int p0, p1, oc0, oc1;
};

void JobFn(void* arg) {
    const Job& j = *static_cast<const Job*>(arg);
    RunOne(j.d, j.in, j.filt, j.out, j.window, j.stage, j.p0, j.p1, j.oc0, j.oc1);
}

bool DualAvailable() {
#if CONV_OPT_DUAL
    return esp_nn_dual_core_active();
#else
    return false;
#endif
}

// Mode im2col: dựng ma trận cửa sổ theo từng khúc nhỏ ngay trong SRAM nội rồi chạy đường 1x1 SIMD của ESP-NN
// trên khúc đó (xem như ảnh 1x1 với wpad kênh). Không cần ma trận lớn trong PSRAM.
// Vùng làm việc của mỗi lõi: [scratch của asm | khúc ma trận], nằm trong buffer kCoreBytes.
struct I2cJob {
    const ConvOptData* d;
    const int8_t* in;
    const int8_t* filt;
    int8_t* out;
    int8_t* work;
    int p0, p1;
};

inline int I2cScratchBytes(int wpad, int out_c) {
    return (24 * wpad + 8 * out_c + 128 + 15) & ~15;
}

void I2cTile(const I2cJob& j) {
    const convopt::Geom& g = j.d->g;
    if (j.p1 <= j.p0) return;
    const int wp = g.wpad;
    const int chunk = j.d->i2c_chunk;
    int8_t* scratch = j.work;
    int8_t* mat = j.work + I2cScratchBytes(wp, g.out_c);
    memset(mat, 0, static_cast<size_t>(chunk) * wp);   // phần đuôi mỗi hàng giữ 0

    int oy = j.p0 / g.out_w;
    int ox = j.p0 - oy * g.out_w;
    for (int p = j.p0; p < j.p1; p += chunk) {
        const int n = (j.p1 - p) < chunk ? (j.p1 - p) : chunk;
        for (int k = 0; k < n; ++k) {
            convopt::BuildWindow(g, j.in, mat + static_cast<size_t>(k) * wp, oy, ox, j.d->q.pad_val);
            if (++ox == g.out_w) {
                ox = 0;
                ++oy;
            }
        }
        esp_nn_conv_s8_mult8_1x1_esp32s3(
            mat, static_cast<uint16_t>(n), 1, static_cast<uint16_t>(wp), j.d->input_offset, j.filt,
            j.d->bias_raw, j.out + static_cast<size_t>(p) * g.out_c, static_cast<uint16_t>(n), 1,
            static_cast<uint16_t>(g.out_c), j.d->q.out_offset, j.d->q.shift, j.d->q.mult, -128, 127, scratch);
    }
}

void I2cJobFn(void* arg) {
    I2cTile(*static_cast<const I2cJob*>(arg));
}

bool RunIm2col(const ConvOptData* d, const int8_t* in, const int8_t* filt, int8_t* out) {
    if (!d->i2c_ok || (reinterpret_cast<uintptr_t>(filt) & 15u) != 0) return false;
    const int pixels = d->g.out_h * d->g.out_w;
    I2cJob main_job = {d, in, filt, out, g_core_buf, 0, pixels};

    if (DualAvailable() && pixels >= 32) {
        uint8_t* ws = static_cast<uint8_t*>(esp_nn_dual_core_scratch(kCoreBytes + 16));
        if (ws != nullptr) {
            const int mid = (pixels / 2) & ~7;   // hai nửa đều là bội 8 (pixels chia hết 8)
            I2cJob worker_job = {d, in, filt, out,
                                 reinterpret_cast<int8_t*>((reinterpret_cast<uintptr_t>(ws) + 15u) & ~uintptr_t(15u)),
                                 mid, pixels};
            if (esp_nn_dual_core_run(I2cJobFn, &worker_job)) {
                main_job.p1 = mid;
                I2cTile(main_job);
                esp_nn_dual_core_wait();
                return true;
            }
        }
    }
    I2cTile(main_job);
    return true;
}

// Chạy theo mode (kOne / kDualOC / kDualPx / kIm2col). Trả false nếu cách này không dùng được
// (kết quả khi đó chưa được ghi).
bool RunMode(const ConvOptData* d, int mode, const int8_t* in, const int8_t* filt, int8_t* out) {
    const convopt::Geom& g = d->g;
    const int pixels = g.out_h * g.out_w;
    int8_t* window = g_core_buf;
    int8_t* stage = d->packed != nullptr ? nullptr : g_core_buf + kWindowBytes;

    if (mode == kIm2col) return RunIm2col(d, in, filt, out);
    if (mode == kOne) {
        RunOne(d, in, filt, out, window, stage, 0, pixels, 0, g.out_c);
        return true;
    }

    if (!DualAvailable()) return false;
    const bool split_oc = (mode == kDualOC);
    if ((split_oc ? g.out_c : pixels) < 2) return false;

    uint8_t* ws = static_cast<uint8_t*>(esp_nn_dual_core_scratch(kCoreBytes + 16));
    if (ws == nullptr) return false;
    int8_t* w_window = reinterpret_cast<int8_t*>((reinterpret_cast<uintptr_t>(ws) + 15u) & ~uintptr_t(15u));
    int8_t* w_stage = d->packed != nullptr ? nullptr : w_window + kWindowBytes;

    // Lõi chính nhận nửa đầu, lõi phụ nhận nửa sau.
    const int mid = (split_oc ? g.out_c : pixels) / 2;
    Job job = {d, in, filt, out, w_window, w_stage, 0, pixels, 0, g.out_c};
    if (split_oc) {
        job.oc0 = mid;
    } else {
        job.p0 = mid;
    }
    if (!esp_nn_dual_core_run(JobFn, &job)) return false;

    if (split_oc) {
        RunOne(d, in, filt, out, window, stage, 0, pixels, 0, mid);
    } else {
        RunOne(d, in, filt, out, window, stage, 0, mid, 0, g.out_c);
    }
    esp_nn_dual_core_wait();
    return true;
}

// ---------------------------------------------------------------- tự đo
TfLiteStatus Tune(TfLiteContext* context, TfLiteNode* node, ConvOptData* d,
                  const int8_t* in, const int8_t* filt, int8_t* out) {
    const int64_t t_start = esp_timer_get_time();
    d->tuned = true;
    d->mode = kBase;

    const size_t bytes = static_cast<size_t>(d->g.out_h) * d->g.out_w * d->g.out_c;

    // Chạy kernel gốc 2 lần (lần đầu làm nóng cache), lấy lần sau làm mốc và kết quả chuẩn.
    TfLiteStatus st = BaseInvoke(context, node, d);
    if (st != kTfLiteOk) return st;
    int64_t t0 = esp_timer_get_time();
    st = BaseInvoke(context, node, d);
    if (st != kTfLiteOk) return st;
    int us[kModeCount] = {static_cast<int>(esp_timer_get_time() - t0), kNA, kNA, kNA, kNA};

    int8_t* ref = static_cast<int8_t*>(heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (ref != nullptr) {
        memcpy(ref, out, bytes);
        for (int m = kOne; m < kModeCount; ++m) {
            memset(out, 0x5A, bytes);
            if (!RunMode(d, m, in, filt, out)) continue;   // kNA
            if (memcmp(out, ref, bytes) != 0) {
                us[m] = kBad;
                continue;
            }
            t0 = esp_timer_get_time();
            RunMode(d, m, in, filt, out);
            us[m] = static_cast<int>(esp_timer_get_time() - t0);
            if (memcmp(out, ref, bytes) != 0) us[m] = kBad;
        }
        // Chọn cách nhanh nhất; chỉ bỏ kernel gốc khi nhanh hơn ít nhất 5%.
        int best = kBase;
        int best_us = us[kBase];
        for (int m = kOne; m < kModeCount; ++m) {
            if (us[m] >= 0 && us[m] < best_us && us[m] * 100 < us[kBase] * 95) {
                best = m;
                best_us = us[m];
            }
        }
        d->mode = best;
        memcpy(out, ref, bytes);
        heap_caps_free(ref);
    }
    ++g_mode_count[d->mode];

#if CONV_OPT_REPORT
    // Đơn vị us. one/oc/px/i2c: -1 = không dùng được, -2 = lệch kết quả so với kernel gốc.
    MicroPrintf("[Tune] #%d s%d %dx%dx%d>%dx%dx%d base=%d one=%d oc=%d px=%d i2c=%d -> %s",
                d->index, d->g.stride, d->g.in_h, d->g.in_w, d->g.in_c,
                d->g.out_h, d->g.out_w, d->g.out_c,
                us[kBase], us[kOne], us[kDualOC], us[kDualPx], us[kIm2col], kModeName[d->mode]);
#endif
    g_tune_us += esp_timer_get_time() - t_start;
    return kTfLiteOk;
}

// ---------------------------------------------------------------- khởi tạo / chuẩn bị
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* data = static_cast<ConvOptData*>(
        context->AllocatePersistentBuffer(context, sizeof(ConvOptData)));
    if (data == nullptr) return nullptr;

    memset(data, 0, sizeof(*data));
    data->base = tflite::Register_CONV_2D();
    data->base_user_data = data->base.init ? data->base.init(context, buffer, length) : nullptr;
    return data;
}

// Kiểm tra hình dạng / tham số; điền Geom. Chưa xét lượng tử hoá.
bool MakeGeom(const TfLiteTensor* input, const TfLiteTensor* filter, const TfLiteTensor* output,
              const TfLiteConvParams& p, convopt::Geom* g) {
    using convopt::kKernel;
    if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 || output->type != kTfLiteInt8) return false;
    if (input->dims->size != 4 || filter->dims->size != 4 || output->dims->size != 4) return false;
    if (input->dims->data[0] != 1 || output->dims->data[0] != 1) return false;

    g->in_h = input->dims->data[1];
    g->in_w = input->dims->data[2];
    g->in_c = input->dims->data[3];
    g->out_h = output->dims->data[1];
    g->out_w = output->dims->data[2];
    g->out_c = output->dims->data[3];

    if (filter->dims->data[0] != g->out_c || filter->dims->data[1] != kKernel ||
        filter->dims->data[2] != kKernel || filter->dims->data[3] != g->in_c) return false;
    if (g->in_c < 1 || g->out_c < 1) return false;

    if (p.stride_width != p.stride_height || (p.stride_width != 1 && p.stride_width != 2)) return false;
    if (p.dilation_width_factor != 1 || p.dilation_height_factor != 1) return false;
    if (p.activation != kTfLiteActNone) return false;
    g->stride = p.stride_width;
    const int s = g->stride;

    if (p.padding == kTfLitePaddingSame) {
        if (g->out_h != (g->in_h + s - 1) / s || g->out_w != (g->in_w + s - 1) / s) return false;
        int ph = (g->out_h - 1) * s + kKernel - g->in_h;
        int pw = (g->out_w - 1) * s + kKernel - g->in_w;
        if (ph < 0) ph = 0;
        if (pw < 0) pw = 0;
        g->pad_top = ph / 2;
        g->pad_left = pw / 2;
    } else if (p.padding == kTfLitePaddingValid) {
        if (g->in_h < kKernel || g->in_w < kKernel) return false;
        if (g->out_h != (g->in_h - kKernel) / s + 1 || g->out_w != (g->in_w - kKernel) / s + 1) return false;
        g->pad_top = 0;
        g->pad_left = 0;
    } else {
        return false;
    }

    g->wlen = kKernel * kKernel * g->in_c;
    g->wpad = (g->wlen + 15) & ~15;
    return g->wpad <= kWindowBytes;
}

TfLiteStatus PrepareFast(TfLiteContext* context, TfLiteNode* node, ConvOptData* data) {
    const auto& params = *static_cast<const TfLiteConvParams*>(node->builtin_data);
    tflite::MicroContext* micro_context = tflite::GetMicroContext(context);

    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
    TfLiteTensor* filter = micro_context->AllocateTempInputTensor(node, 1);
    if (input == nullptr || filter == nullptr || output == nullptr) return kTfLiteError;

    TfLiteStatus status = kTfLiteOk;
    const TfLiteEvalTensor* filter_eval = tflite::micro::GetEvalInput(context, node, 1);
    const int8_t* w = filter_eval ? tflite::micro::GetTensorData<int8_t>(filter_eval) : nullptr;

    convopt::Geom g = {};
    bool ok = w != nullptr && MakeGeom(input, filter, output, params, &g) &&
              filter->quantization.type == kTfLiteAffineQuantization &&
              filter->quantization.params != nullptr && EnsureCoreBuffer();

    // Filter đối xứng (zero_point = 0), lượng tử hoá theo kênh (hoặc 1 giá trị).
    const TfLiteAffineQuantization* affine = nullptr;
    int nscale = 0;
    if (ok) {
        affine = static_cast<const TfLiteAffineQuantization*>(filter->quantization.params);
        nscale = affine->scale->size;
        const int nzp = affine->zero_point->size;
        if ((nscale != 1 && nscale != g.out_c) || (nzp != 1 && nzp != g.out_c)) {
            ok = false;
        } else {
            for (int oc = 0; oc < g.out_c; ++oc) {
                if (affine->zero_point->data[nzp == 1 ? 0 : oc] != 0) { ok = false; break; }
            }
        }
    }

    int32_t* buf = nullptr;
    int8_t* packed = nullptr;
    if (ok) {
        buf = static_cast<int32_t*>(
            context->AllocatePersistentBuffer(context, 4 * g.out_c * sizeof(int32_t)));
        if (buf == nullptr) {
            status = kTfLiteError;
            ok = false;
        }
    }
    if (ok && g.wpad != g.wlen) {
        // Độ dài cửa sổ không chia hết 16 (conv đầu, Cin=3): chép filter sang bản có đuôi 0,
        // giữ ở SRAM nội vì chỉ vài trăm byte.
        packed = static_cast<int8_t*>(heap_caps_aligned_alloc(
            16, static_cast<size_t>(g.out_c) * g.wpad, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
        if (packed == nullptr) {
            ok = false;
        } else {
            memset(packed, 0, static_cast<size_t>(g.out_c) * g.wpad);
            for (int oc = 0; oc < g.out_c; ++oc) {
                memcpy(packed + static_cast<size_t>(oc) * g.wpad, w + static_cast<size_t>(oc) * g.wlen, g.wlen);
            }
        }
    }

    if (ok) {
        const int32_t input_offset = -input->params.zero_point;
        const int32_t* bias = nullptr;
        if (node->inputs->size == 3 && node->inputs->data[2] >= 0) {
            const TfLiteEvalTensor* bias_eval = tflite::micro::GetEvalInput(context, node, 2);
            bias = bias_eval ? tflite::micro::GetTensorData<int32_t>(bias_eval) : nullptr;
        }

        data->q.mult = buf;
        data->q.shift = buf + g.out_c;
        data->q.corr = buf + 2 * g.out_c;
        data->q.out_offset = output->params.zero_point;
        data->q.pad_val = static_cast<int8_t>(-input_offset);
        int32_t* mult = buf;
        int32_t* shift = buf + g.out_c;
        int32_t* corr = buf + 2 * g.out_c;

        for (int oc = 0; oc < g.out_c; ++oc) {
            const double real_multiplier =
                static_cast<double>(input->params.scale) *
                static_cast<double>(affine->scale->data[nscale == 1 ? 0 : oc]) /
                static_cast<double>(output->params.scale);

            int32_t quantized_multiplier = 0;
            int sh = 0;
            tflite::QuantizeMultiplier(real_multiplier, &quantized_multiplier, &sh);
            mult[oc] = quantized_multiplier;
            shift[oc] = sh;

            int32_t filter_sum = 0;
            const int8_t* wo = w + static_cast<size_t>(oc) * g.wlen;
            for (int i = 0; i < g.wlen; ++i) filter_sum += wo[i];
            corr[oc] = (bias != nullptr ? bias[oc] : 0) + filter_sum * input_offset;
        }

        // bias gốc (không cộng phần zero_point) cho đường 1x1 của ESP-NN
        int32_t* bias_raw = buf + 3 * g.out_c;
        for (int oc = 0; oc < g.out_c; ++oc) bias_raw[oc] = bias != nullptr ? bias[oc] : 0;
        data->bias_raw = bias_raw;
        data->input_offset = input_offset;

        // Mode im2col: số điểm ảnh phải chia hết 8 (đường dư của asm 1x1 chậm hơn nhiều) và một khúc
        // (>= 8 điểm ảnh) phải nằm vừa cùng scratch của asm trong buffer SRAM của mỗi lõi.
        data->i2c_ok = false;
        data->i2c_chunk = 0;
        if (((g.out_h * g.out_w) % 8) == 0 && (g.wpad % 8) == 0 && g.out_c <= 512) {
            int chunk = (kCoreBytes - I2cScratchBytes(g.wpad, g.out_c)) / g.wpad;
            if (chunk * g.wpad > 12288) chunk = 12288 / g.wpad;   // khúc <= 12 KB cho gọn
            chunk &= ~7;
            if (chunk >= 8) {
                data->i2c_chunk = chunk;
                data->i2c_ok = true;
            }
        }

        data->g = g;
        data->packed = packed;
        data->eligible = true;
        ++g_eligible;
    }

    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(filter);
    return status;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    if (node == nullptr || node->builtin_data == nullptr || node->user_data == nullptr) {
        return kTfLiteError;
    }
    auto* data = static_cast<ConvOptData*>(node->user_data);
    data->index = ++g_node_seq;
    data->i2c_ok = false;
    data->i2c_chunk = 0;
    data->eligible = false;
    data->tuned = false;
    data->mode = kBase;

    // Luôn chuẩn bị kernel gốc: nó là đường dự phòng và là mốc để so sánh khi tự đo.
    node->user_data = data->base_user_data;
    TfLiteStatus status = data->base.prepare ? data->base.prepare(context, node) : kTfLiteOk;
    node->user_data = data;
    if (status != kTfLiteOk) return status;

    return PrepareFast(context, node, data);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
    if (node == nullptr || node->user_data == nullptr) return kTfLiteError;
    auto* data = static_cast<ConvOptData*>(node->user_data);

    const bool need_tune = g_tune && !data->tuned;
    if (!data->eligible || (!need_tune && data->mode == kBase)) {
        return BaseInvoke(context, node, data);
    }

    const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
    const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);
    TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
    const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
    const int8_t* filt = data->packed != nullptr ? data->packed : tflite::micro::GetTensorData<int8_t>(filter);
    int8_t* out = tflite::micro::GetTensorData<int8_t>(output);

    if (need_tune) return Tune(context, node, data, in, filt, out);

    if (!RunMode(data, data->mode, in, filt, out)) {
        // Hai lõi không còn dùng được: quay về một lõi (kết quả giống hệt).
        RunMode(data, kOne, in, filt, out);
    }
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration RegisterConvOpt() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

// Danh sách shape cố định không còn dùng: mọi conv 3x3 đủ điều kiện đều được tự đo.
void ConvOptSetTargets(const ConvOptTarget* targets, int count) {
    (void)targets;
    (void)count;
}

// Số conv 3x3 đủ điều kiện chạy bằng kernel ConvOpt (trước khi tự đo).
int ConvOptFastNodeCount() {
    return g_eligible;
}

int ConvOptStagingBytes() {
    return g_core_buf != nullptr ? kCoreBytes : 0;
}

void ConvOptTuneBegin() {
    g_tune = true;
    g_tune_us = 0;
    for (int m = 0; m < kModeCount; ++m) g_mode_count[m] = 0;
}

void ConvOptTuneEnd() {
    g_tune = false;
}

void ConvOptPrintSummary() {
    MicroPrintf("[ConvOpt] %d conv 3x3 | base=%d one=%d dual-oc=%d dual-px=%d im2col=%d | tune %d ms",
                g_eligible, g_mode_count[kBase], g_mode_count[kOne], g_mode_count[kDualOC],
                g_mode_count[kDualPx], g_mode_count[kIm2col], static_cast<int>(g_tune_us / 1000));
}
