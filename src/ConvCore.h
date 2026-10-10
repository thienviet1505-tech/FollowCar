#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Phần tính toán thuần C++ (không phụ thuộc TFLite Micro / ESP-NN) của kernel conv 3x3 theo từng
// điểm ảnh (ConvOpt). Tách riêng để kiểm thử trên PC. Hàm dot và phép nhân lượng tử hoá được
// truyền vào dưới dạng functor để firmware dùng bản ESP32-S3, PC dùng bản C thuần.
namespace convopt {

constexpr int kKernel = 3;

struct Geom {
    int in_h, in_w, in_c;
    int out_h, out_w, out_c;
    int stride;              // 1 hoặc 2 (dùng chung cho hai chiều)
    int pad_top, pad_left;   // đệm ngầm (SAME); VALID thì bằng 0
    int wlen;                // 9 * in_c
    int wpad;                // wlen làm tròn lên bội của 16 (độ dài thật của mỗi hàng filter)
};

struct Quant {
    const int32_t* mult;     // [out_c]
    const int32_t* shift;    // [out_c]
    const int32_t* corr;     // [out_c] = bias + tổng filter * input_offset
    int32_t out_offset;      // zero_point của output
    int8_t pad_val;          // zero_point của input (giá trị điền ô ngoài biên)
};

// Một cửa sổ 3x3xin_c quanh điểm ra (oy, ox); ô ngoài biên điền pad_val.
// Chỉ ghi window[0, wlen); phần đuôi [wlen, wpad) do RunTile xoá về 0.
inline void BuildWindow(const Geom& g, const int8_t* input, int8_t* window,
                        int oy, int ox, int8_t pad_val) {
    const int in_c = g.in_c;
    const int row_stride = g.in_w * in_c;
    const int row_bytes = kKernel * in_c;
    const int y0 = oy * g.stride - g.pad_top;
    const int x0 = ox * g.stride - g.pad_left;

    for (int fy = 0; fy < kKernel; ++fy) {
        const int iy = y0 + fy;
        int8_t* dst = window + fy * row_bytes;

        if (iy < 0 || iy >= g.in_h) {
            memset(dst, pad_val, row_bytes);
            continue;
        }
        if (x0 >= 0 && x0 + kKernel <= g.in_w) {
            memcpy(dst, input + static_cast<size_t>(iy) * row_stride + static_cast<size_t>(x0) * in_c,
                   row_bytes);
            continue;
        }
        for (int fx = 0; fx < kKernel; ++fx) {
            const int ix = x0 + fx;
            if (ix < 0 || ix >= g.in_w) {
                memset(dst + fx * in_c, pad_val, in_c);
            } else {
                memcpy(dst + fx * in_c,
                       input + static_cast<size_t>(iy) * row_stride + static_cast<size_t>(ix) * in_c, in_c);
            }
        }
    }
}

// Tính ô vuông [p0, p1) x [oc0, oc1) của đầu ra (p = oy * out_w + ox). Các ô khác nhau ghi vào
// vùng output rời nhau nên hai lõi có thể chạy hai ô song song.
//   filter : out_c hàng, mỗi hàng wpad byte (đuôi [wlen, wpad) bằng 0), địa chỉ căn 16.
//   window : >= wpad byte, căn 16.
//   stage  : vùng đệm SRAM căn 16 (stage_bytes byte) để chép filter theo khối kênh ra, mỗi filter
//            chỉ đọc 1 lần từ PSRAM; nullptr thì đọc thẳng từ filter (dùng khi filter đã ở SRAM).
//   dot(a, b, len) = tổng a[i]*b[i] (a, b căn 16, len bội 16); req(acc, mult, shift) = nhân lượng tử hoá.
template <class Dot, class Req>
void RunTile(const Geom& g, const Quant& q, const int8_t* input, const int8_t* filter, int8_t* output,
             int8_t* window, int8_t* stage, size_t stage_bytes,
             int p0, int p1, int oc0, int oc1, Dot dot, Req req) {
    if (p0 >= p1 || oc0 >= oc1) return;
    if (g.wpad > g.wlen) memset(window + g.wlen, 0, g.wpad - g.wlen);

    int chunk = oc1 - oc0;
    if (stage != nullptr) {
        const int fit = static_cast<int>(stage_bytes / static_cast<size_t>(g.wpad));
        if (fit >= 1) {
            if (fit < chunk) chunk = fit;
        } else {
            stage = nullptr;
        }
    }

    for (int c0 = oc0; c0 < oc1; c0 += chunk) {
        const int n = (oc1 - c0) < chunk ? (oc1 - c0) : chunk;
        const int8_t* w = filter + static_cast<size_t>(c0) * g.wpad;
        if (stage != nullptr) {
            memcpy(stage, w, static_cast<size_t>(n) * g.wpad);
            w = stage;
        }

        int oy = p0 / g.out_w;
        int ox = p0 - oy * g.out_w;
        for (int p = p0; p < p1; ++p) {
            BuildWindow(g, input, window, oy, ox, q.pad_val);

            int8_t* out = output + static_cast<size_t>(p) * g.out_c + c0;
            for (int i = 0; i < n; ++i) {
                int32_t acc = dot(window, w + static_cast<size_t>(i) * g.wpad, g.wpad);
                acc += q.corr[c0 + i];
                acc = req(acc, q.mult[c0 + i], q.shift[c0 + i]);
                acc += q.out_offset;
                acc = acc < -128 ? -128 : (acc > 127 ? 127 : acc);
                out[i] = static_cast<int8_t>(acc);
            }

            if (++ox == g.out_w) {
                ox = 0;
                ++oy;
            }
        }
    }
}

}  // namespace convopt
