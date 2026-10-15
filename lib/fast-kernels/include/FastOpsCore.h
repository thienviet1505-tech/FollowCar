#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Phần tính toán thuần C++ (không phụ thuộc TFLite Micro) của các kernel PadFast, TransposeFast,
// SoftmaxLut. Tách riêng để kiểm thử trên PC so với kết quả của interpreter.
namespace fastops {

// ---------------------------------------------------------------- PAD
// Đệm không gian cho tensor NHWC int8: copy từng hàng bằng memcpy, phần đệm bằng memset.
inline void PadSpatialS8(const int8_t* in, int8_t* out, int n, int h, int w, int c,
                         int top, int bottom, int left, int right, int8_t pad_value) {
    const int out_w = w + left + right;
    const size_t in_row = static_cast<size_t>(w) * c;
    const size_t out_row = static_cast<size_t>(out_w) * c;
    const size_t left_bytes = static_cast<size_t>(left) * c;
    const size_t right_bytes = static_cast<size_t>(right) * c;
    for (int b = 0; b < n; ++b) {
        for (int y = 0; y < top; ++y) {
            memset(out, pad_value, out_row);
            out += out_row;
        }
        for (int y = 0; y < h; ++y) {
            if (left_bytes) memset(out, pad_value, left_bytes);
            memcpy(out + left_bytes, in, in_row);
            if (right_bytes) memset(out + left_bytes + in_row, pad_value, right_bytes);
            in += in_row;
            out += out_row;
        }
        for (int y = 0; y < bottom; ++y) {
            memset(out, pad_value, out_row);
            out += out_row;
        }
    }
}

// ---------------------------------------------------------------- TRANSPOSE
constexpr int kMaxTransposeRank = 6;

// Kế hoạch duyệt theo thứ tự output: dims[k] là kích thước, stride[k] là bước nhảy trong input.
// Các chiều kích thước 1 được bỏ, các chiều liền nhau mà cũng liền nhau trong input được gộp.
struct TransposePlan {
    int rank;
    int dims[kMaxTransposeRank];
    int stride[kMaxTransposeRank];
};

// in_dims[rank] là shape input, perm[rank] là hoán vị (output dim k lấy từ input dim perm[k]).
inline bool MakeTransposePlan(const int* in_dims, int rank, const int* perm, TransposePlan* plan) {
    if (rank < 1 || rank > kMaxTransposeRank) return false;
    int in_stride[kMaxTransposeRank];
    int s = 1;
    for (int d = rank - 1; d >= 0; --d) {
        in_stride[d] = s;
        s *= in_dims[d];
    }
    bool seen[kMaxTransposeRank] = {false};
    int r = 0;
    for (int k = 0; k < rank; ++k) {
        const int p = perm[k];
        if (p < 0 || p >= rank || seen[p]) return false;
        seen[p] = true;
        if (in_dims[p] == 1) continue;
        if (r > 0 && plan->stride[r - 1] == in_stride[p] * in_dims[p]) {
            // gộp vào chiều đứng trước: chiều trước nay có thêm extent của chiều này
            plan->dims[r - 1] *= in_dims[p];
            plan->stride[r - 1] = in_stride[p];
            continue;
        }
        plan->dims[r] = in_dims[p];
        plan->stride[r] = in_stride[p];
        ++r;
    }
    plan->rank = r;
    return true;
}

inline void TransposeS8(const TransposePlan& plan, const int8_t* in, int8_t* out) {
    const int r = plan.rank;
    if (r == 0) {  // chỉ còn 1 phần tử (mọi chiều bằng 1)
        out[0] = in[0];
        return;
    }
    if (r == 1) {
        const int n = plan.dims[0], st = plan.stride[0];
        if (st == 1) {
            memcpy(out, in, n);
        } else {
            for (int i = 0; i < n; ++i) out[i] = in[static_cast<size_t>(i) * st];
        }
        return;
    }
    if (r == 2 && plan.dims[1] < 16 && plan.stride[0] == 1) {
        // Dạng NCHW -> NHWC với ít kênh (vd. 3): đọc tuần tự từng mặt phẳng, ghi cách quãng.
        const int d0 = plan.dims[0], d1 = plan.dims[1], s1 = plan.stride[1];
        for (int j = 0; j < d1; ++j) {
            const int8_t* src = in + static_cast<size_t>(j) * s1;
            int8_t* dst = out + j;
            for (int i = 0; i < d0; ++i) {
                *dst = src[i];
                dst += d1;
            }
        }
        return;
    }

    const int outer = r - 1;
    const int n = plan.dims[outer], st = plan.stride[outer];
    size_t outer_total = 1;
    for (int d = 0; d < outer; ++d) outer_total *= plan.dims[d];
    int idx[kMaxTransposeRank] = {0};
    size_t base = 0;
    for (size_t it = 0; it < outer_total; ++it) {
        if (st == 1) {
            memcpy(out, in + base, n);
        } else {
            const int8_t* src = in + base;
            for (int i = 0; i < n; ++i) out[i] = src[static_cast<size_t>(i) * st];
        }
        out += n;
        for (int d = outer - 1; d >= 0; --d) {
            base += plan.stride[d];
            if (++idx[d] < plan.dims[d]) break;
            base -= static_cast<size_t>(plan.stride[d]) * plan.dims[d];
            idx[d] = 0;
        }
    }
}

// ---------------------------------------------------------------- SOFTMAX int8 -> int8 (trục cuối)
// tab[d] = exp(-beta * scale_in * d), d = (max - x) trong 0..255.
inline void BuildSoftmaxTable(float* tab, float beta, float scale_in) {
    for (int d = 0; d < 256; ++d) {
        tab[d] = static_cast<float>(exp(-static_cast<double>(beta) * scale_in * d));
    }
}

inline void SoftmaxRowsS8(const int8_t* in, int8_t* out, int rows, int depth, const float* tab,
                          float inv_scale_out, int zero_point_out) {
    for (int r = 0; r < rows; ++r) {
        int mx = in[0];
        for (int i = 1; i < depth; ++i) mx = in[i] > mx ? in[i] : mx;
        float sum = 0.0f;
        for (int i = 0; i < depth; ++i) sum += tab[mx - in[i]];
        const float k = inv_scale_out / sum;
        for (int i = 0; i < depth; ++i) {
            int q = static_cast<int>(tab[mx - in[i]] * k + 0.5f) + zero_point_out;  // giá trị >= 0 nên +0.5 là làm tròn
            out[i] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
        in += depth;
        out += depth;
    }
}

}  // namespace fastops
