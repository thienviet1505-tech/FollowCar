#include "Preprocess.h"
#include <Arduino.h>
#include <algorithm>
#include "Letterbox.h"

namespace {
constexpr int kChannels = 3;               // RGB
constexpr int kLetterboxPadPixel = 114;    // màu xám đệm chuẩn của YOLO (114,114,114)
}  // namespace

bool Preprocess::frameToInput(const camera_fb_t* fb, ModelInference& model) {
    const int modelW = model.inputWidth();
    const int modelH = model.inputHeight();
    int8_t* outBuf = model.inputData();
    const int8_t* lut = model.pixelLut();

    const size_t needed = (size_t)modelW * modelH * kChannels;
    if (model.inputBytes() < needed) {
        Serial.println("[Preprocess] Buffer dau ra qua nho.");
        return false;
    }

    const uint16_t* src = (const uint16_t*)fb->buf;
    const int srcW = fb->width;
    const int srcH = fb->height;
    const Letterbox lb = Letterbox::make(srcW, srcH, modelW, modelH);
    const bool nhwc = model.inputIsNhwc();

    const size_t planeSize = (size_t)modelW * modelH;
    std::fill(outBuf, outBuf + needed, lut[kLetterboxPadPixel]);

    for (int y = 0; y < lb.resizeH; y++) {
        const int dstY = y + lb.padY;
        if (dstY < 0 || dstY >= modelH) continue;

        int srcY = (int)((int64_t)y * srcH / lb.resizeH);
        if (srcY >= srcH) srcY = srcH - 1;

        for (int x = 0; x < lb.resizeW; x++) {
            const int dstX = x + lb.padX;
            if (dstX < 0 || dstX >= modelW) continue;

            int srcX = (int)((int64_t)x * srcW / lb.resizeW);
            if (srcX >= srcW) srcX = srcW - 1;

            uint16_t px = src[srcY * srcW + srcX];
            px = __builtin_bswap16(px);

            const uint8_t r5 = (px >> 11) & 0x1F;
            const uint8_t g6 = (px >> 5) & 0x3F;
            const uint8_t b5 = px & 0x1F;

            const uint8_t r8 = (r5 << 3) | (r5 >> 2);
            const uint8_t g8 = (g6 << 2) | (g6 >> 4);
            const uint8_t b8 = (b5 << 3) | (b5 >> 2);

            const size_t pixelIdx = (size_t)dstY * modelW + dstX;

            if (nhwc) {
                int8_t* dst = outBuf + pixelIdx * 3;
                dst[0] = lut[r8];
                dst[1] = lut[g8];
                dst[2] = lut[b8];
            } else {
                outBuf[0 * planeSize + pixelIdx] = lut[r8];
                outBuf[1 * planeSize + pixelIdx] = lut[g8];
                outBuf[2 * planeSize + pixelIdx] = lut[b8];
            }
        }
    }
    return true;
}
