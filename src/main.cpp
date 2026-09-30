#include <Arduino.h>
#include <vector>
#include <cmath>
#include "CameraModule.h"
#include "ModelInference.h"
#include "Detector.h"
#include "SerialLogger.h"
#include "ViewMode.h"

namespace {
CameraModule camera;
ModelInference model;

// Input model là float32.
// Chỉ chuẩn hoá pixel 8-bit về [0, 1].
inline float quantizePixel(uint8_t pixel8, float scale, int zeroPoint) {
    (void)scale;
    (void)zeroPoint;
    return pixel8 / 255.0f;
}

// Resize nearest-neighbor từ framebuffer RGB565 xuống đúng kích thước input.
// Model sử dụng layout NCHW: [1, 3, H, W].
bool preprocessFrame(camera_fb_t* fb, float* outBuf, size_t outBufSize,
                     int modelW, int modelH, int modelC) {
    if (modelC != 3) {
        Serial.println("[Preprocess] Chi ho tro model 3 kenh (RGB) o khung nay.");
        return false;
    }

    size_t needed = (size_t)modelW * modelH * modelC;
    if (outBufSize < needed) {
        Serial.println("[Preprocess] Buffer dau ra qua nho.");
        return false;
    }

    float scale = model.inputScale();
    int zeroPoint = model.inputZeroPoint();

    const uint16_t* src = (const uint16_t*)fb->buf;
    int srcW = fb->width;
    int srcH = fb->height;

    size_t planeSize = (size_t)modelW * modelH;

    for (int y = 0; y < modelH; y++) {
        int srcY = y * srcH / modelH;

        for (int x = 0; x < modelW; x++) {
            int srcX = x * srcW / modelW;

            uint16_t px = src[srcY * srcW + srcX];
            // px = __builtin_bswap16(px);  // bỏ comment nếu màu bị đảo

            uint8_t r5 = (px >> 11) & 0x1F;
            uint8_t g6 = (px >> 5) & 0x3F;
            uint8_t b5 = px & 0x1F;

            uint8_t r8 = (r5 << 3) | (r5 >> 2);
            uint8_t g8 = (g6 << 2) | (g6 >> 4);
            uint8_t b8 = (b5 << 3) | (b5 >> 2);

            size_t pixelIdx = (size_t)y * modelW + x;

            // NCHW: [R plane][G plane][B plane]
            outBuf[0 * planeSize + pixelIdx] = quantizePixel(r8, scale, zeroPoint);
            outBuf[1 * planeSize + pixelIdx] = quantizePixel(g8, scale, zeroPoint);
            outBuf[2 * planeSize + pixelIdx] = quantizePixel(b8, scale, zeroPoint);
        }
    }

    return true;
}
}  // namespace

void setup() {
    SerialLogger::begin(115200);
    delay(500);
    Serial.println("Booting...");

    if (!camera.begin()) {
        Serial.println("Camera init that bai, dung lai.");
        while (true) delay(1000);
    }

    if (!model.begin()) {
        Serial.println("Model init that bai, dung lai.");
        while (true) delay(1000);
    }

    Serial.println("Setup xong, bat dau vong lap detect.");
    Serial.println("Go ? de xem cac lenh (v = xem anh camera, d = ve nhan dien).");
}

void loop() {
    ViewMode::handleSerial(camera);
    if (ViewMode::active()) {  // dang xem anh tren trinh duyet -> tam dung nhan dien
        delay(50);
        return;
    }

    Serial.println("[Loop] truoc capture");

    camera_fb_t* fb = camera.capture();

    Serial.println("[Loop] sau capture");

    if (!fb) {
        Serial.println("Capture that bai, bo qua frame.");
        delay(100);
        return;
    }

    int origW = fb->width;
    int origH = fb->height;

    static std::vector<float> inputBuf(
        model.inputBytes() / sizeof(float));

    Serial.println("[Loop] bat dau preprocess");
    
    bool ok = preprocessFrame(
        fb,
        inputBuf.data(),
        inputBuf.size(),
        model.inputWidth(),
        model.inputHeight(),
        model.inputChannels()
    );

    Serial.println("[Loop] ket thuc preprocess");

    camera.release(fb);

    if (!ok) {
        Serial.println("Preprocess that bai.");
        delay(50);
        return;
    }

    Serial.println("[Loop] bat dau invoke");

    if (!model.invoke(
        inputBuf.data(),
        inputBuf.size() * sizeof(float)
    )) {
        Serial.println("Inference that bai.");
        delay(50);
        return;
    }

    Serial.println("[Loop] ket thuc invoke");
    
    std::vector<Detection> detections;
    Detector::decode(model, origW, origH, detections);
    SerialLogger::logFrameResult(detections, origW, origH);

    delay(50);
}