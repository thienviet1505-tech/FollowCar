#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include "CameraModule.h"
#include "ModelInference.h"
#include "Detector.h"
#include "SerialLogger.h"
#include "ViewMode.h"
#include "Robot.h"
#include "RobotConfig.h"
#include "RobotMath.h"
#include "test_sample.h"

bool g_triggerSelfTest = false;

// Layout input do ModelInference xác định lúc nạp model (định nghĩa ở phạm vi toàn cục trong ModelInference.cpp).
// Phải khai báo NGOÀI namespace ẩn danh bên dưới, nếu không linker tìm nhầm ký hiệu trong namespace.
extern bool g_model_input_nhwc;

namespace {
CameraModule camera;
ModelInference model;

// Trung bình trượt 5 frame gần nhất cho tổng thời gian xử lý và tuổi frame.
struct Avg5 {
    float v[5] = {0, 0, 0, 0, 0};
    int n = 0, idx = 0;
    void add(float x) { v[idx] = x; idx = (idx + 1) % 5; if (n < 5) n++; }
    float get() const { float s = 0; for (int i = 0; i < n; i++) s += v[i]; return n ? s / n : 0.0f; }
};
Avg5 avgTotal, avgAge, avgLoop;
uint32_t frameId = 0;
int64_t prevFrameStartUs = 0;   // 0 = chưa có frame trước (để tính chu kỳ thực của loop)


// Resize nearest-neighbor từ framebuffer RGB565 xuống đúng kích thước input.
// Model sử dụng layout NCHW: [1, 3, H, W]. outBuf là tensor input int8,
// lut là bảng đổi pixel 0..255 -> int8 (ModelInference::pixelLut()).
bool preprocessFrame(camera_fb_t* fb, int8_t* outBuf, size_t outBufSize, const int8_t* lut,
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

    const uint16_t* src = (const uint16_t*)fb->buf;
    int srcW = fb->width;
    int srcH = fb->height;

    // Letterbox: giữ nguyên tỉ lệ ảnh, sau đó padding về modelW x modelH.
    float scaleW = (float)modelW / (float)srcW;
    float scaleH = (float)modelH / (float)srcH;
    float scale = std::min(scaleW, scaleH);

    int resizeW = (int)std::round(srcW * scale);
    int resizeH = (int)std::round(srcH * scale);

    float padXf = (modelW - resizeW) / 2.0f;
    float padYf = (modelH - resizeH) / 2.0f;

    // Cách làm tròn gần với Ultralytics LetterBox.
    int padX = (int)std::round(padXf - 0.1f);
    int padY = (int)std::round(padYf - 0.1f);

    size_t planeSize = (size_t)modelW * modelH;

    // Ultralytics dùng padding màu 114.
    const int8_t padValue = lut[114];

    // Ban đầu toàn bộ canvas là padding.
    std::fill(outBuf, outBuf + needed, padValue);

    // Vẽ ảnh đã resize vào giữa canvas 224x224.
    for (int y = 0; y < resizeH; y++) {
        int dstY = y + padY;
        if (dstY < 0 || dstY >= modelH) continue;

        int srcY = (int)((int64_t)y * srcH / resizeH);
        if (srcY >= srcH) srcY = srcH - 1;

        for (int x = 0; x < resizeW; x++) {
            int dstX = x + padX;
            if (dstX < 0 || dstX >= modelW) continue;

            int srcX = (int)((int64_t)x * srcW / resizeW);
            if (srcX >= srcW) srcX = srcW - 1;

            uint16_t px = src[srcY * srcW + srcX];
            px = __builtin_bswap16(px);

            uint8_t r5 = (px >> 11) & 0x1F;
            uint8_t g6 = (px >> 5) & 0x3F;
            uint8_t b5 = px & 0x1F;

            uint8_t r8 = (r5 << 3) | (r5 >> 2);
            uint8_t g8 = (g6 << 2) | (g6 >> 4);
            uint8_t b8 = (b5 << 3) | (b5 >> 2);

            size_t pixelIdx = (size_t)dstY * modelW + dstX;

            if (g_model_input_nhwc) {
                // NHWC: [R G B] liền nhau cho từng điểm ảnh
                int8_t* dst = outBuf + pixelIdx * 3;
                dst[0] = lut[r8];
                dst[1] = lut[g8];
                dst[2] = lut[b8];
            } else {
                // NCHW: [R plane][G plane][B plane]
                outBuf[0 * planeSize + pixelIdx] = lut[r8];
                outBuf[1 * planeSize + pixelIdx] = lut[g8];
                outBuf[2 * planeSize + pixelIdx] = lut[b8];
            }

        }
    }

    return true;
}

// Chay kiem tra model bang anh mau (to tien 100k, 224x224) duoc nhung san trong flash.
void runSelfTest(ModelInference& model) {
    Serial.println();
    Serial.println("================ SELF-TEST ================");

    if (TEST_SAMPLE_SIZE != model.inputBytes()) {
        Serial.printf("[SelfTest] FAIL | anh mau %u byte != input model %u byte\n",
                      (unsigned)TEST_SAMPLE_SIZE, (unsigned)model.inputBytes());
        Serial.println("============================================");
        return;
    }
    int8_t* in = model.inputData();
    const int8_t* lut = model.pixelLut();
    if (g_model_input_nhwc) {
        // Ảnh mẫu lưu NCHW; model NHWC: đổi chỗ khi nạp.
        const size_t plane = (size_t)TEST_SAMPLE_W * TEST_SAMPLE_H;
        for (size_t p = 0; p < plane; p++) {
            for (int c = 0; c < 3; c++) {
                in[p * 3 + c] = lut[pgm_read_byte(&TEST_SAMPLE_NCHW[c * plane + p])];
            }
        }
    } else {
        for (size_t i = 0; i < TEST_SAMPLE_SIZE; i++) {
            in[i] = lut[pgm_read_byte(&TEST_SAMPLE_NCHW[i])];
        }
    }

    const uint32_t t0 = millis();
    const bool ok = model.invoke();
    const uint32_t invokeMs = millis() - t0;

    if (!ok) {
        Serial.printf("[SelfTest] FAIL | invoke=%lums\n", (unsigned long)invokeMs);
        Serial.println("============================================");
        return;
    }

    std::vector<Detection> detections;
    Detector::decode(model, TEST_SAMPLE_W, TEST_SAMPLE_H, detections);

    const std::vector<float>& mx = Detector::classMaxScores();
    const float score100k = mx.empty() ? 0.0f : mx[0];
    const bool detected = score100k >= 0.3f;

    Serial.printf("[SelfTest] invoke=%lums | 100k=%.3f | result=%s\n",
                  (unsigned long)invokeMs, score100k, detected ? "PASS" : "FAIL");
    Serial.println("============================================");
}
}  // namespace

void setup() {
    SerialLogger::begin(115200);
    delay(500);
    Serial.println("[Boot] start");

    if (!camera.begin()) {
        Serial.println("[Boot] camera init FAIL");
        while (true) delay(1000);
    }

    if (!model.begin()) {
        Serial.println("[Boot] model init FAIL");
        while (true) delay(1000);
    }

    runSelfTest(model);
    camera.discardFrames(3);
    Robot::begin();   // cảm biến + động cơ + task điều khiển (robot phải đứng yên ~1.5 s)
    Serial.println("[Boot] detect ready");

    Serial.printf("[Boot] internal free=%u largest=%u\n",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

void loop() {
    ViewMode::handleSerial(camera);
    if (g_triggerSelfTest) {
        g_triggerSelfTest = false;
        runSelfTest(model);
        camera.discardFrames(3);
        prevFrameStartUs = 0;
        return;
    }
    if (ViewMode::active()) {
        prevFrameStartUs = 0;
        // Ở chế độ xem ảnh xe thả trôi: in encoder mỗi 1 giây để lăn tay hiệu chỉnh GEAR_RATIO / kiểm tra dấu.
        static uint32_t lastViewPrintMs = 0;
        if (millis() - lastViewPrintMs >= 1000) {
            lastViewPrintMs = millis();
            Robot::printFrameLine();
        }
        delay(50);
        return;
    }

    const uint32_t currentFrame = ++frameId;
    const int64_t tFrame0 = esp_timer_get_time();
    // Chu kỳ thực của loop = từ lúc bắt đầu frame trước đến lúc bắt đầu frame này (đã gồm thời gian in Serial).
    if (prevFrameStartUs) avgLoop.add((tFrame0 - prevFrameStartUs) / 1000.0f);
    prevFrameStartUs = tFrame0;
    // Dòng ngăn cách ngắn (1 dòng ~20 byte ~ 2ms; bản cũ ~80 byte ~ 7ms ở 115200 baud)
    Serial.printf("==== Frame %lu ====\n", (unsigned long)currentFrame);

    const int64_t tLoop0 = esp_timer_get_time();   // pipeline đo từ đây (không tính thời gian in)
    const int64_t tCap0 = esp_timer_get_time();
    camera_fb_t* fb = camera.capture();
    const int64_t tCap1 = esp_timer_get_time();

    if (!fb) {
        Serial.println("[Capture] FAIL");
        delay(100);
        return;
    }

    const float ageMs = (tCap1 - ((int64_t)fb->timestamp.tv_sec * 1000000LL + fb->timestamp.tv_usec)) / 1000.0f;
    const int origW = fb->width;
    const int origH = fb->height;

    const int64_t tPre0 = esp_timer_get_time();
    const bool ok = preprocessFrame(fb, model.inputData(), model.inputBytes(), model.pixelLut(),
                                    model.inputWidth(), model.inputHeight(), model.inputChannels());
    const int64_t tPre1 = esp_timer_get_time();
    camera.release(fb);

    if (!ok) {
        Serial.println("[Preprocess] FAIL");
        delay(50);
        return;
    }

    const int64_t tInv0 = esp_timer_get_time();
    if (!model.invoke()) {
        Serial.println("[Inference] FAIL");
        delay(50);
        return;
    }
    const int64_t tInv1 = esp_timer_get_time();

    std::vector<Detection> detections;
    const int64_t tDec0 = esp_timer_get_time();
    Detector::decode(model, origW, origH, detections);
    const int64_t tDec1 = esp_timer_get_time();

    // Chọn tờ TARGET_CLASS lớn nhất, báo cho tầng điều khiển (góc lệch + thời điểm chụp để bù độ trễ).
    const Detection* target = nullptr;
    for (const Detection& d : detections) {
        if (d.class_id == Cfg::TARGET_CLASS && (!target || d.area() > target->area())) target = &d;
    }
    Robot::onDetection(target != nullptr, target ? RM::bearingDeg((float)target->cx, (float)origW) : 0.0f,
                       tCap1 - (int64_t)(ageMs * 1000.0f));

    // Đo đến đây để không tính thời gian in Serial vào hiệu năng pipeline.
    const float totalMs = (tDec1 - tLoop0) / 1000.0f;
    const float captureMs = (tCap1 - tCap0) / 1000.0f;
    const float preMs = (tPre1 - tPre0) / 1000.0f;
    const float invokeMs = (tInv1 - tInv0) / 1000.0f;
    const float decodeMs = (tDec1 - tDec0) / 1000.0f;

    avgTotal.add(totalMs);
    avgAge.add(ageMs);

    // FPS_pipe = chỉ tính chụp+tiền xử lý+suy luận+giải mã. FPS_loop = chu kỳ thực của cả vòng loop (có cả thời gian in Serial).
    const float fpsLoop = avgLoop.n ? 1000.0f / avgLoop.get() : 0.0f;
    Serial.printf("[Perf] cap=%.0fms pre=%.0fms infer=%.0fms dec=%.0fms | total=%.0fms avg5=%.0fms FPS_pipe=%.2f age=%.0fms | FPS_loop=%.2f\n",
                  captureMs, preMs, invokeMs, decodeMs,
                  totalMs, avgTotal.get(), 1000.0f / avgTotal.get(), avgAge.get(), fpsLoop);
    SerialLogger::logFrameResult(detections, origW, origH);
    Robot::printFrameLine();
}