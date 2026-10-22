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
extern bool g_model_input_nhwc;

namespace {
CameraModule camera;
ModelInference model;

struct Avg5 {
    float v[5] = {0, 0, 0, 0, 0};
    int n = 0, idx = 0;
    void add(float x) { v[idx] = x; idx = (idx + 1) % 5; if (n < 5) n++; }
    float get() const { float s = 0; for (int i = 0; i < n; i++) s += v[i]; return n ? s / n : 0.0f; }
};
Avg5 avgTotal, avgAge, avgLoop;
uint32_t frameId = 0;
int64_t prevFrameStartUs = 0;
bool s_holdingTarget = false;   // trạng thái giữ mục tiêu cho ngưỡng kép (Hysteresis)

const char* className(int id) {
    static const char* const kNames[] = {"100k", "10k", "50k", "5k"};
    return (id >= 0 && id < 4) ? kNames[id] : "?";
}

void printDetectionsLog(const std::vector<Detection>& detections, int origW, int origH) {
    if (detections.empty()) {
        Serial.println("[Detect] none");
        return;
    }
    uint32_t classMask = 0;
    const Detection* largest = &detections[0];
    float maxArea = largest->w * largest->h;
    for (const auto& d : detections) {
        if (d.class_id >= 0 && d.class_id < 32) classMask |= (1u << d.class_id);
        const float area = d.w * d.h;
        if (area > maxArea) {
            maxArea = area;
            largest = &d;
        }
    }
    int numClasses = 0;
    for (uint32_t m = classMask; m; m >>= 1) numClasses += (m & 1u);

    const int boxW = (int)std::lround(largest->w);
    const int boxH = (int)std::lround(largest->h);
    const int offX = (int)std::lround(largest->cx - origW * 0.5f);
    const int offY = (int)std::lround(largest->cy - origH * 0.5f);

    Serial.printf("[Detect] n=%u | classes=%d | largest=%s score=%.3f box=%dx%d offset=(%+d,%+d)px\n",
                  (unsigned)detections.size(), numClasses, className(largest->class_id),
                  largest->score, boxW, boxH, offX, offY);
}

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

    float scaleW = (float)modelW / (float)srcW;
    float scaleH = (float)modelH / (float)srcH;
    float scale = std::min(scaleW, scaleH);

    int resizeW = (int)std::round(srcW * scale);
    int resizeH = (int)std::round(srcH * scale);

    float padXf = (modelW - resizeW) / 2.0f;
    float padYf = (modelH - resizeH) / 2.0f;

    int padX = (int)std::round(padXf - 0.1f);
    int padY = (int)std::round(padYf - 0.1f);

    size_t planeSize = (size_t)modelW * modelH;
    const int8_t padValue = lut[114];
    std::fill(outBuf, outBuf + needed, padValue);

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
    Robot::begin();
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
        s_holdingTarget = false;
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
    if (prevFrameStartUs) avgLoop.add((tFrame0 - prevFrameStartUs) / 1000.0f);
    prevFrameStartUs = tFrame0;
    Serial.printf("==== Frame %lu ====\n", (unsigned long)currentFrame);

    const int64_t tLoop0 = esp_timer_get_time();
    const int64_t tCap0 = esp_timer_get_time();
    camera_fb_t* fb = camera.capture();
    const int64_t tCap1 = esp_timer_get_time();

    if (!fb) {
        Serial.println("[Camera] capture FAIL");
        Robot::onDetection(false, 0.0f, 0.0f, tCap1);
        Robot::printFrameLine();
        delay(10);
        return;
    }

    const int origW = fb->width;
    const int origH = fb->height;
    const int64_t frameTimeUs = (fb->timestamp.tv_sec > 0)
        ? ((int64_t)fb->timestamp.tv_sec * 1000000LL + (int64_t)fb->timestamp.tv_usec)
        : ((tCap0 + tCap1) / 2);

    const int64_t tPre0 = esp_timer_get_time();
    const bool preOk = preprocessFrame(
        fb, model.inputData(), model.inputBytes(), model.pixelLut(),
        TEST_SAMPLE_W, TEST_SAMPLE_H, 3
    );
    camera.release(fb);
    const int64_t tPre1 = esp_timer_get_time();

    if (!preOk) {
        Robot::onDetection(false, 0.0f, 0.0f, frameTimeUs);
        Robot::printFrameLine();
        return;
    }

    const int64_t tInf0 = esp_timer_get_time();
    const bool infOk = model.invoke();
    const int64_t tInf1 = esp_timer_get_time();

    if (!infOk) {
        Serial.println("[NN] invoke FAIL");
        Robot::onDetection(false, 0.0f, 0.0f, frameTimeUs);
        Robot::printFrameLine();
        return;
    }

    const int64_t tDec0 = esp_timer_get_time();
    std::vector<Detection> detections;
    Detector::decode(model, origW, origH, detections);
    const int64_t tDec1 = esp_timer_get_time();

    const float capMs   = (tCap1 - tCap0) / 1000.0f;
    const float preMs   = (tPre1 - tPre0) / 1000.0f;
    const float inferMs = (tInf1 - tInf0) / 1000.0f;
    const float decMs   = (tDec1 - tDec0) / 1000.0f;
    const float totalMs = (tDec1 - tLoop0) / 1000.0f;
    const float ageMs   = (tDec1 - frameTimeUs) / 1000.0f;

    avgTotal.add(totalMs);
    avgAge.add(ageMs);

    const float avg5Ms  = avgTotal.get();
    const float fpsPipe = (avg5Ms > 0.0f) ? (1000.0f / avg5Ms) : 0.0f;
    const float loopMs  = avgLoop.get();
    const float fpsLoop = (loopMs > 0.0f) ? (1000.0f / loopMs) : 0.0f;

    Serial.printf("[Perf] cap=%.0fms pre=%.0fms infer=%.0fms dec=%.0fms | total=%.0fms avg5=%.0fms FPS_pipe=%.2f age=%.0fms | FPS_loop=%.2f\n",
                  capMs, preMs, inferMs, decMs, totalMs, avg5Ms, fpsPipe, avgAge.get(), fpsLoop);

    const std::vector<float>& mx = Detector::classMaxScores();
    if (mx.size() >= 4) {
        Serial.printf("[Score] max: 100k=%.3f 10k=%.3f 50k=%.3f 5k=%.3f\n",
                      mx[0], mx[1], mx[2], mx[3]);
    }

    printDetectionsLog(detections, origW, origH);

    // ---- Chọn mục tiêu thuộc TARGET_CLASS với ngưỡng kép (Hysteresis) ----
    const float minConf = s_holdingTarget ? Cfg::DETECT_HOLD_THRESH : Cfg::DETECT_ACQUIRE_THRESH;
    const Detection* bestTarget = nullptr;
    float bestArea = -1.0f;

    for (const auto& d : detections) {
        if (d.class_id == Cfg::TARGET_CLASS && d.score >= minConf) {
            const float area = d.w * d.h;
            if (area > bestArea) {
                bestArea = area;
                bestTarget = &d;
            }
        }
    }

    if (bestTarget) {
        s_holdingTarget = true;
        const float brg = RM::bearingDeg(bestTarget->cx, (float)origW);
        const float maxSide = std::max(bestTarget->w, bestTarget->h);
        const float boxRatio = (origW > 0) ? (maxSide / (float)origW) : 0.0f;
        Robot::onDetection(true, brg, boxRatio, frameTimeUs);
    } else {
        s_holdingTarget = false;
        Robot::onDetection(false, 0.0f, 0.0f, frameTimeUs);
    }

    Robot::printFrameLine();
}