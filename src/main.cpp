#include <Arduino.h>
#include <vector>
#include <cmath>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include "CameraModule.h"
#include "ModelInference.h"
#include "Detector.h"
#include "Preprocess.h"
#include "TargetSelect.h"
#include "SerialLogger.h"
#include "BleLog.h"
#include "ViewMode.h"
#include "Robot.h"
#include "RobotConfig.h"
#include "test_sample.h"

namespace {
CameraModule camera;
ModelInference model;
TargetSelector s_selector;      // chọn mục tiêu: ngưỡng kép (Hysteresis), cổng (gate), làm mượt điểm (xem TargetSelect.h)

// ---- Thống kê thời gian ----
struct Avg5 {
    float v[5] = {0, 0, 0, 0, 0};
    int n = 0, idx = 0;
    void add(float x) { v[idx] = x; idx = (idx + 1) % 5; if (n < 5) n++; }
    float get() const { float s = 0; for (int i = 0; i < n; i++) s += v[i]; return n ? s / n : 0.0f; }
};
Avg5 avgTotal, avgAge, avgLoop;
uint32_t frameId = 0;
int64_t prevFrameStartUs = 0;

// Mốc thời gian (us) của một frame
struct FrameTimes {
    int64_t loop0 = 0, cap0 = 0, cap1 = 0, pre0 = 0, pre1 = 0, inf0 = 0, inf1 = 0, dec0 = 0, dec1 = 0;
};

struct FrameStats {
    float capMs = 0, preMs = 0, inferMs = 0, decMs = 0, totalMs = 0;
    float avg5Ms = 0, fpsPipe = 0, ageAvgMs = 0, fpsLoop = 0;
};

// ---- Khởi động / camera watchdog ----
int s_bootReason = 0;
const char* s_bootReasonName = "";
int s_camFails = 0;                          // số lần capture lỗi liên tiếp
bool s_camRebooted = false;                  // lần boot này là do xe tự khởi động lại vì camera kẹt
constexpr uint32_t kCamRebootMagic = 0xCA3E5A11u;
RTC_NOINIT_ATTR uint32_t s_rebootWhy;        // sống sót qua esp_restart(); rác sau khi cấp nguồn lại nên chỉ tin khi lý do reset = SW

// ---- Log ----
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
                  (unsigned)detections.size(), numClasses, Cfg::className(largest->class_id),
                  largest->score, boxW, boxH, offX, offY);
}

// Điểm cao nhất của từng class trong frame vừa decode (kể cả dưới ngưỡng) — xem model đang "nghĩ" gì.
void printClassScores() {
    const std::vector<float>& mx = Detector::classMaxScores();
    if (mx.empty()) return;
    Serial.print("[Score] max:");
    for (size_t c = 0; c < mx.size(); c++) {
        if ((int)c < Cfg::NUM_CLASS_NAMES) Serial.printf(" %s=%.3f", Cfg::CLASS_NAMES[c], mx[c]);
        else Serial.printf(" c%u=%.3f", (unsigned)c, mx[c]);
    }
    Serial.println();
}

// Một dòng tóm tắt cờ build + giờ build: sau mỗi lần đổi cờ và nạp lại, nhìn dòng này (Serial/BLE) là biết xe đang chạy bản nào.
const char* flagsLine() {
    static char buf[200];
    snprintf(buf, sizeof buf,
             "[Cfg] %s %s SPD=%.2f BLE=%d VIEW=%d SLEW=%d FF0=%d IMU=%d TOF=%d CAM=%d MED=%d FUSE=%d STOP=%d BTN=%d ZUPT=%d GRACE=%d SLOG=%d TEST=%d",
             __DATE__, __TIME__, Cfg::SPEED_SCALE, ROBOT_BLE_LOG, ROBOT_VIEW_MODE,
             (int)Cfg::PWM_SLEW_LIMIT, (int)Cfg::FF_SMOOTH_ZERO, (int)Cfg::IMU_WATCHDOG, (int)Cfg::TOF_WATCHDOG,
             (int)Cfg::CAMERA_WATCHDOG, (int)Cfg::TOF_MEDIAN_FILTER, (int)Cfg::TOF_FUSE_DISTANCE, (int)Cfg::TOF_SAFETY_STOP, (int)Cfg::BUTTON_ENABLE,
             (int)Cfg::GYRO_ZUPT, (int)Cfg::REACQUIRE_GRACE, (int)Cfg::SERIAL_FRAME_LOG, (int)Cfg::BOOT_SELFTEST);
    return buf;
}

// ---- Self-test: chạy model trên ảnh mẫu cố định, báo PASS nếu thấy 100k ----
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
    if (model.inputIsNhwc()) {
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

// ---- Khởi động ----

// BROWNOUT = tụt nguồn; PANIC/WDT = treo chương trình
const char* bootReasonName(esp_reset_reason_t rr) {
    if (rr == ESP_RST_BROWNOUT) return " (BROWNOUT)";
    if (rr == ESP_RST_PANIC) return " (PANIC)";
    if (rr == ESP_RST_TASK_WDT || rr == ESP_RST_INT_WDT || rr == ESP_RST_WDT) return " (WDT)";
    return "";
}

// Ghi lại lý do reset lần này, và xem có phải lần trước xe tự khởi động lại vì camera kẹt không.
void noteBootReason() {
    const esp_reset_reason_t rr = esp_reset_reason();
    s_bootReason = (int)rr;
    s_camRebooted = (rr == ESP_RST_SW && s_rebootWhy == kCamRebootMagic);
    s_rebootWhy = 0;
    if (s_camRebooted) Serial.println("[Boot] Lan truoc xe TU KHOI DONG LAI vi camera loi lien tiep");
    s_bootReasonName = bootReasonName(rr);
    Serial.printf("[Boot] reset reason=%d%s\n", s_bootReason, s_bootReasonName);
}

[[noreturn]] void haltBoot(const char* what) {
    Serial.printf("[Boot] %s init FAIL\n", what);
    while (true) delay(1000);
}

// ---- Một vòng nhận diện ----

// Báo cho Robot biết frame này không có mục tiêu (lỗi camera/tiền xử lý/model).
void publishMiss(int64_t frameTimeUs) {
    Robot::onDetection(false, 0.0f, 0.0f, frameTimeUs);
    Robot::printFrameLine();
}

void onCaptureFail(int64_t tCap1) {
    Serial.println("[Camera] capture FAIL");
    BleLog::log("! camera capture FAIL (%d)", s_camFails + 1);
    // Camera kẹt (nhiễu/sụt áp làm mất đồng bộ): mỗi lần esp_camera_fb_get chờ hàng giây rồi trả null, xe không có nhận diện mới
    // nên đứng yên. Không có cách hồi phục nhẹ -> khởi động lại; lý do được ghi lại ở lần boot sau.
    if (Cfg::CAMERA_WATCHDOG && ++s_camFails >= Cfg::CAMERA_FAIL_REBOOT_COUNT) {
        Serial.println("[Camera] loi lien tiep -> khoi dong lai");
        BleLog::log("! camera loi lien tiep -> khoi dong lai");
        s_rebootWhy = kCamRebootMagic;
        delay(400);   // cho BLE kịp gửi dòng cuối
        ESP.restart();
    }
    publishMiss(tCap1);
    delay(10);
}

FrameStats updateStats(const FrameTimes& t, int64_t frameTimeUs) {
    FrameStats s;
    s.capMs   = (t.cap1 - t.cap0) / 1000.0f;
    s.preMs   = (t.pre1 - t.pre0) / 1000.0f;
    s.inferMs = (t.inf1 - t.inf0) / 1000.0f;
    s.decMs   = (t.dec1 - t.dec0) / 1000.0f;
    s.totalMs = (t.dec1 - t.loop0) / 1000.0f;
    const float ageMs = (t.dec1 - frameTimeUs) / 1000.0f;

    avgTotal.add(s.totalMs);
    avgAge.add(ageMs);

    s.avg5Ms = avgTotal.get();
    s.fpsPipe = (s.avg5Ms > 0.0f) ? (1000.0f / s.avg5Ms) : 0.0f;
    s.ageAvgMs = avgAge.get();
    const float loopMs = avgLoop.get();
    s.fpsLoop = (loopMs > 0.0f) ? (1000.0f / loopMs) : 0.0f;
    return s;
}

void printFrameLog(const FrameStats& s, const std::vector<Detection>& detections, int origW, int origH) {
    Serial.printf("[Perf] cap=%.0fms pre=%.0fms infer=%.0fms dec=%.0fms | total=%.0fms avg5=%.0fms FPS_pipe=%.2f age=%.0fms | FPS_loop=%.2f\n",
                  s.capMs, s.preMs, s.inferMs, s.decMs, s.totalMs, s.avg5Ms, s.fpsPipe, s.ageAvgMs, s.fpsLoop);
    printClassScores();
    printDetectionsLog(detections, origW, origH);
}

// Chọn mục tiêu thuộc TARGET_CLASS (ngưỡng kép, cổng, làm mượt điểm: xem TargetSelect.h) rồi báo cho Robot.
void publishTarget(const std::vector<Detection>& detections, int origW, int64_t frameTimeUs) {
    const TargetSelector::Pick pick = s_selector.select(detections, origW, Robot::state().yawDeg);
    if (pick.found) {
        Robot::onDetection(true, pick.bearingDeg, pick.boxRatio, frameTimeUs);
    } else {
        Robot::onDetection(false, 0.0f, 0.0f, frameTimeUs);
    }
}

void detectFrame() {
    const uint32_t currentFrame = ++frameId;
    const int64_t tFrame0 = esp_timer_get_time();
    if (prevFrameStartUs) avgLoop.add((tFrame0 - prevFrameStartUs) / 1000.0f);
    prevFrameStartUs = tFrame0;
    if (Cfg::SERIAL_FRAME_LOG) Serial.printf("==== Frame %lu ====\n", (unsigned long)currentFrame);

    FrameTimes t;
    t.loop0 = esp_timer_get_time();
    t.cap0 = esp_timer_get_time();
    camera_fb_t* fb = camera.capture();
    t.cap1 = esp_timer_get_time();

    if (!fb) {
        onCaptureFail(t.cap1);
        return;
    }
    s_camFails = 0;

    const int origW = fb->width;
    const int origH = fb->height;
    const int64_t frameTimeUs = (fb->timestamp.tv_sec > 0)
        ? ((int64_t)fb->timestamp.tv_sec * 1000000LL + (int64_t)fb->timestamp.tv_usec)
        : ((t.cap0 + t.cap1) / 2);

    t.pre0 = esp_timer_get_time();
    const bool preOk = Preprocess::frameToInput(fb, model);
    camera.release(fb);
    t.pre1 = esp_timer_get_time();

    if (!preOk) {
        publishMiss(frameTimeUs);
        return;
    }

    t.inf0 = esp_timer_get_time();
    const bool infOk = model.invoke();
    t.inf1 = esp_timer_get_time();

    if (!infOk) {
        Serial.println("[NN] invoke FAIL");
        BleLog::log("! NN invoke FAIL");
        publishMiss(frameTimeUs);
        return;
    }

    t.dec0 = esp_timer_get_time();
    std::vector<Detection> detections;
    Detector::decode(model, origW, origH, detections);
    t.dec1 = esp_timer_get_time();

    const FrameStats stats = updateStats(t, frameTimeUs);
    if (Cfg::SERIAL_FRAME_LOG) printFrameLog(stats, detections, origW, origH);

    publishTarget(detections, origW, frameTimeUs);
    Robot::printFrameLine();

    if (BleLog::enabled()) {
        const std::vector<float>& mx = Detector::classMaxScores();
        BleLog::log("P fps=%.1f age=%.0f inf=%.0f sc=%.2f free=%u", stats.fpsPipe, stats.ageAvgMs, stats.inferMs,
                    (int)mx.size() > Cfg::TARGET_CLASS ? mx[Cfg::TARGET_CLASS] : 0.0f,
                    (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    }
}

// Đang ở chế độ VIEW (xem ảnh qua WiFi): tạm dừng nhận diện, mỗi giây in một dòng trạng thái.
void idleInViewMode() {
    prevFrameStartUs = 0;
    s_selector.reset();
    static uint32_t lastViewPrintMs = 0;
    if (millis() - lastViewPrintMs >= 1000) {
        lastViewPrintMs = millis();
        Robot::printFrameLine();
    }
    delay(50);
}

}  // namespace

void setup() {
    SerialLogger::begin(115200);
    delay(500);
    Serial.println("[Boot] start");
    noteBootReason();
    Serial.println(flagsLine());
    if (!Cfg::TOF_SAFETY_STOP) Serial.println("[Cfg] CANH BAO: TOF_SAFETY_STOP = false -> xe KHONG tu khoa chieu tien khi ToF thay vat can");

    if (!camera.begin()) haltBoot("camera");
    if (!model.begin()) haltBoot("model");

    if (Cfg::BOOT_SELFTEST) runSelfTest(model);
    else Serial.println("[Boot] bo qua self-test (BOOT_SELFTEST = false)");
    camera.discardFrames(3);
    Robot::begin();
    Serial.println("[Boot] detect ready");

    Serial.printf("[Boot] internal free=%u largest=%u\n",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    // Để CUỐI CÙNG: camera/model lấy bộ nhớ trước, BLE (nếu ROBOT_BLE_LOG = 1) lấy phần còn lại và bắt đầu phát.
    BleLog::begin();
    BleLog::log("[Boot] reset reason=%d%s", s_bootReason, s_bootReasonName);
    if (s_camRebooted) BleLog::log("[Boot] lan truoc xe tu khoi dong lai vi camera loi lien tiep");
    BleLog::log("%s", flagsLine());
    if (!Cfg::TOF_SAFETY_STOP) BleLog::log("[Cfg] CANH BAO: TOF_SAFETY_STOP = false");
}

void loop() {
    ViewMode::handleSerial(camera);
    if (ViewMode::takeSelfTestRequest()) {
        runSelfTest(model);
        camera.discardFrames(3);
        prevFrameStartUs = 0;
        return;
    }
    if (ViewMode::active()) {
        idleInViewMode();
        return;
    }
    detectFrame();
}
