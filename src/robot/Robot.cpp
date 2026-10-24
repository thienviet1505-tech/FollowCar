#include "Robot.h"
#include <Arduino.h>
#include <esp_timer.h>
#include "RobotConfig.h"
#include "RobotMath.h"
#include "Sensors.h"
#include "Drive.h"
#include "BleLog.h"

using namespace Cfg;

namespace {

using Robot::Mode;

struct Det {
    uint32_t seq = 0;
    bool seen = false;
    float bearing = 0;
    float boxRatio = 0;
    int64_t frameUs = 0;
};

portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
Det s_det;                          // loop() -> task điều khiển
Robot::State s_state;               // task điều khiển -> loop()
volatile bool s_enabled = true;     // đặt bởi ViewMode (v/d)
volatile uint32_t s_btnCount = 0, s_btnTaken = 0;

const char* modeName(Mode m) {
    switch (m) {
        case Mode::CALIB: return "CALIB";
        case Mode::VIEW:  return "VIEW";
        case Mode::LOST:  return "LOST";
        case Mode::TRACK: return "TRACK";
        default:          return "BLOCK";
    }
}

void controlTask(void*) {
    Sensors::Reading rd;
    RM::WheelPid pidL, pidR;
    RM::YawHistory hist;
    RM::YawHistory odoHist;     // lịch sử quãng đường (mm), dùng lại cấu trúc của YawHistory

    Mode mode = Mode::CALIB;
    uint32_t modeSince = millis();
    uint32_t nextCalibMs = 0;
    bool coasted = false;
    bool warnedNoImu = false;

    uint32_t lastDetSeq = 0;
    uint32_t lastSeenMs = 0;
    int seenStreak = 0;
    float yawTarget = 0;
    bool hasYawTarget = false;
    float filtBoxRatio = 0;
    float visBaseMm = 0, visBaseOdo = 0;       // khoảng cách ước lượng từ box tại lúc chụp frame + quãng đường lúc đó
    bool hasVis = false;
    bool holdDist = false;                     // đang ở vùng dừng 15cm (trễ Schmitt trong followLaw)
    uint16_t visDistMm = TOF_NONE_MM;
    bool dispSeen = false;
    float dispBearing = 0;

    float vL = 0, vR = 0, tgtL = 0, tgtR = 0;
    // Cửa sổ encoder SPEED_WINDOW_TICKS chu kỳ để đo vận tốc bánh (giảm nhiễu lượng tử hóa 1 xung/chu kỳ)
    constexpr int W = SPEED_WINDOW_TICKS + 1;
    int32_t wEncL[W] = {0}, wEncR[W] = {0};
    int64_t wT[W] = {0};
    int wHead = 0, wCount = 0;
    auto resetSpeedWindow = [&]() { wHead = 0; wCount = 0; vL = vR = 0; };
    int pwmL = 0, pwmR = 0;
    float winMax = 0, dtShown = 0;
    int winTicks = 0;

    bool btnArmed = false, btnFired = false;
    uint32_t btnLowSince = 0;
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    int btnRaw = digitalRead(BUTTON_PIN), btnRawPrev = btnRaw;
    uint32_t btnPresses = 0;

    Sensors::update(0.0f, rd);
    resetSpeedWindow();

    TickType_t last = xTaskGetTickCount();
    int64_t tPrev = esp_timer_get_time();

    auto setMode = [&](Mode m) {
        if (m == mode) return;
        Serial.printf("[Robot] %s -> %s\n", modeName(mode), modeName(m));
        BleLog::log("MODE %s->%s", modeName(mode), modeName(m));
        mode = m;
        modeSince = millis();
        coasted = false;
        // Chỉ xóa PID (dừng cứng) khi vào VIEW/CALIB. Với LOST/TRACK/BLOCKED giữ đà để giảm tốc/đổi tốc êm.
        if (m == Mode::VIEW || m == Mode::CALIB) {
            pidL.reset();
            pidR.reset();
        }
        tgtL = tgtR = 0;
        if (m == Mode::LOST || m == Mode::VIEW || m == Mode::CALIB) {
            hasYawTarget = false;
            hasVis = false;
            holdDist = false;
            filtBoxRatio = 0;
            visDistMm = TOF_NONE_MM;
        }
    };

    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(CTRL_PERIOD_MS));
        const int64_t now = esp_timer_get_time();
        const uint32_t nowMs = millis();
        const float dt = (float)(now - tPrev) * 1e-6f;
        tPrev = now;
        const float dtc = dt < 0.05f ? dt : 0.05f;

        const float dtMs = dt * 1e3f;
        if (dtMs > winMax) winMax = dtMs;
        if (++winTicks >= 200) { dtShown = winMax; winMax = 0; winTicks = 0; }

        // ---- cảm biến + vận tốc bánh ----
        Sensors::update(dt < 0.1f ? dt : 0.1f, rd);
        hist.push(now, rd.yawDeg);
        const float odo = RM::countsToMm(0.5f * (float)(rd.encL + rd.encR));   // quay tại chỗ thì 2 bánh triệt tiêu nhau
        odoHist.push(now, odo);
        wEncL[wHead] = rd.encL;
        wEncR[wHead] = rd.encR;
        wT[wHead] = now;
        wHead = (wHead + 1) % W;
        if (wCount < W) wCount++;
        if (wCount >= 2) {
            const int newest = (wHead - 1 + W) % W;
            const int oldest = (wHead - wCount + W) % W;
            const float span = (float)(wT[newest] - wT[oldest]) * 1e-6f;
            if (span > 0.0f) {
                vL = RM::countsPerSecToMmPerSec((float)(wEncL[newest] - wEncL[oldest]) / span);
                vR = RM::countsPerSecToMmPerSec((float)(wEncR[newest] - wEncR[oldest]) / span);
            }
        }

        // ---- nút G0 ----
        btnRaw = digitalRead(BUTTON_PIN);
        const bool btnLow = (btnRaw == LOW);
        if (btnRaw != btnRawPrev) {
            Serial.println(btnLow ? "[Robot] G0 xuong" : "[Robot] G0 len");
            BleLog::log("G0 %s", btnLow ? "xuong" : "len");
            btnRawPrev = btnRaw;
        }
        if (!btnLow) {
            btnArmed = true;
            btnFired = false;
            btnLowSince = 0;
        } else if (btnArmed && !btnFired) {
            if (btnLowSince == 0) btnLowSince = nowMs;
            if (nowMs - btnLowSince >= (uint32_t)BUTTON_DEBOUNCE_MS) {
                btnFired = true;
                btnPresses++;
                s_enabled = false;
                portENTER_CRITICAL(&s_mux);
                s_btnCount++;
                portEXIT_CRITICAL(&s_mux);
                Serial.println("[Robot] Nut G0 -> dung xe");
                BleLog::log("G0 -> dung xe");
            }
        }

        // ---- kết quả nhận diện mới từ loop() ----
        Det det;
        portENTER_CRITICAL(&s_mux);
        det = s_det;
        portEXIT_CRITICAL(&s_mux);
        if (det.seq != lastDetSeq) {
            lastDetSeq = det.seq;
            dispSeen = det.seen;
            dispBearing = det.bearing;
            if (det.seen) {
                seenStreak++;
                lastSeenMs = nowMs;
                // Hướng tuyệt đối của tiền (tại lúc chụp frame), lọc mượt giữa các frame
                const float z = hist.at(det.frameUs, rd.yawDeg) + det.bearing;
                if (!hasYawTarget) {
                    yawTarget = z;
                    hasYawTarget = true;
                } else {
                    yawTarget += YAW_TARGET_FILTER_ALPHA * (z - yawTarget);
                }
                // Khoảng cách từ box, trộn với giá trị cũ đã trừ quãng đường xe đi được từ lúc đó đến lúc chụp frame
                const uint16_t meas = RM::estimateVisualDistMm(det.boxRatio);
                if (meas < TOF_NONE_MM) {
                    const float odoF = DIST_DEAD_RECKON ? odoHist.at(det.frameUs, odo) : 0.0f;
                    if (hasVis) {
                        const float pred = visBaseMm - (DIST_DEAD_RECKON ? odoF - visBaseOdo : 0.0f);
                        visBaseMm = pred + BOX_RATIO_FILTER_ALPHA * ((float)meas - pred);
                    } else {
                        visBaseMm = (float)meas;
                        hasVis = true;
                    }
                    visBaseMm = RM::clampf(visBaseMm, 50.0f, (float)TOF_MAX_MM);
                    visBaseOdo = odoF;
                    filtBoxRatio = (float)FOLLOW_DISTANCE_MM * BOX_RATIO_AT_FOLLOW_DIST / visBaseMm;
                }
            } else {
                // Giảm dần seenStreak thay vì xóa trắng ngay nếu mới hụt 1 frame
                if (seenStreak > 0) seenStreak--;
            }
        }

        // Khoảng cách hiện tại = khoảng cách lúc chụp - quãng đường đã đi từ đó (không chạy theo số cũ khi hụt frame)
        if (hasVis) {
            visDistMm = (uint16_t)RM::clampf(visBaseMm - (DIST_DEAD_RECKON ? odo - visBaseOdo : 0.0f),
                                             50.0f, (float)TOF_MAX_MM);
        }

        // ---- máy trạng thái ----
        const bool enabled = s_enabled;
        const bool tofFresh = Sensors::tofPresent() && rd.tofAgeMs <= TOF_STALE_MS;

        if (mode == Mode::CALIB) {
            if (!Sensors::imuPresent()) {
                if (!warnedNoImu) { Serial.println("[Robot] KHONG CO IMU -> xe se khong chay"); warnedNoImu = true; }
            } else if (nowMs >= nextCalibMs) {
                Serial.println("[Robot] Hieu chinh gyro 1.5 s - giu xe DUNG YEN");
                if (Sensors::calibrateGyro()) {
                    hist.clear();
                    odoHist.clear();
                    Sensors::update(0.0f, rd);
                    resetSpeedWindow();
                    seenStreak = 0;
                    setMode(enabled ? Mode::LOST : Mode::VIEW);
                } else {
                    nextCalibMs = millis() + GYRO_RETRY_MS;
                }
                tPrev = esp_timer_get_time();
                last = xTaskGetTickCount();
            }
        } else if (!enabled) {
            setMode(Mode::VIEW);
        } else if (mode == Mode::VIEW) {
            seenStreak = 0;
            setMode(Mode::LOST);
        }

        if (mode == Mode::LOST) {
            if (seenStreak >= REACQUIRE_FRAMES && (nowMs - lastSeenMs) <= LOST_TIMEOUT_MS && rd.imuGood) {
                setMode(Mode::TRACK);
            }
        } else if (mode == Mode::TRACK || mode == Mode::BLOCKED) {
            if (!rd.imuGood) {
                Serial.println("[Robot] IMU loi -> dung");
                BleLog::log("IMU loi -> dung");
                seenStreak = 0;
                setMode(Mode::LOST);
            } else if (nowMs - lastSeenMs > LOST_TIMEOUT_MS) {
                seenStreak = 0;
                setMode(Mode::LOST);
            } else if (mode == Mode::TRACK && tofFresh && rd.tofMm < TOF_STOP_MM) {
                setMode(Mode::BLOCKED);
            } else if (mode == Mode::BLOCKED && (!tofFresh || rd.tofMm >= TOF_RESUME_MM)) {
                setMode(Mode::TRACK);
            }
        }

        // ---- đầu ra điều khiển ----
        if (mode == Mode::TRACK || mode == Mode::BLOCKED) {
            RM::FollowCmd c = RM::followLaw(yawTarget - rd.yawDeg, rd.rateDps, rd.tofMm, tofFresh, visDistMm, holdDist);
            // Nếu ToF báo có vật cản sát (< TOF_STOP_MM), khóa chiều tiến (nhưng vẫn cho phép xoay hướng theo tiền)
            if (mode == Mode::BLOCKED) {
                c.v = fminf(c.v, 0.0f);
            }
            RM::wheelTargets(c, tgtL, tgtR);
            pidL.target = tgtL;
            pidR.target = tgtR;
            float fl, fr;
            const bool bl = pidL.step(vL, dtc, fl);
            const bool br = pidR.step(vR, dtc, fr);
            pwmL = bl ? 0 : (int)lroundf(fl);
            pwmR = br ? 0 : (int)lroundf(fr);
            Drive::setPwm(pwmL, pwmR, !COAST_AT_ZERO);
        } else if (mode == Mode::LOST) {
            // Mất tiền: giảm tốc êm theo ACCEL_MM_S2 (đích = 0)
            tgtL = tgtR = 0;
            pidL.target = 0;
            pidR.target = 0;
            float fl, fr;
            const bool bl = pidL.step(vL, dtc, fl);
            const bool br = pidR.step(vR, dtc, fr);
            pwmL = bl ? 0 : (int)lroundf(fl);
            pwmR = br ? 0 : (int)lroundf(fr);
            Drive::setPwm(pwmL, pwmR, !COAST_AT_ZERO);
        } else {
            tgtL = tgtR = 0;
            pwmL = pwmR = 0;
            if (mode == Mode::VIEW && !coasted && (nowMs - modeSince) >= (uint32_t)BRAKE_HOLD_MS) {
                Drive::coast();
                coasted = true;
            } else if (!coasted) {
                Drive::setPwm(0, 0);
            }
        }

        // ---- đăng trạng thái ----
        Robot::State s;
        s.mode = mode;
        s.yawDeg = rd.yawDeg; s.rateDps = rd.rateDps; s.yawTargetDeg = yawTarget;
        s.imuGood = rd.imuGood;
        s.seen = dispSeen; s.bearingDeg = dispBearing;
        s.tofMm = rd.tofMm; s.visMm = visDistMm; s.boxRatio = filtBoxRatio; s.tofAgeMs = rd.tofAgeMs;
        s.encL = rd.encL; s.encR = rd.encR;
        s.vL = vL; s.vR = vR; s.tgtL = tgtL; s.tgtR = tgtR;
        s.pwmL = pwmL; s.pwmR = pwmR;
        s.ctlDtMaxMs = dtShown;
        s.btnLevel = btnRaw;
        s.btnPresses = btnPresses;
        portENTER_CRITICAL(&s_mux);
        s_state = s;
        portEXIT_CRITICAL(&s_mux);
    }
}

}  // namespace

namespace Robot {

void begin() {
    Drive::begin();
    Sensors::begin();
    if (!Sensors::tofPresent()) {
        Serial.println("[Robot] CANH BAO: khong co ToF -> dung khoang cach uoc luong tu Camera BBox");
    }
    xTaskCreatePinnedToCore(controlTask, "robotctl", CTRL_TASK_STACK, nullptr, CTRL_TASK_PRIO, nullptr, CTRL_CORE);
    Serial.printf("[Robot] Task dieu khien %d Hz, loi %d, uu tien %d | lop tien=%d | dung cach %u mm\n",
                  1000 / CTRL_PERIOD_MS, CTRL_CORE, CTRL_TASK_PRIO, TARGET_CLASS, (unsigned)FOLLOW_DISTANCE_MM);
    Serial.printf("[Robot] GEAR_RATIO=%.1f V_AT_PWM_MAX=%.0f PWM_DEADBAND=%.0f HFOV=%.0f\n",
                  GEAR_RATIO, V_AT_PWM_MAX_MM_S, PWM_DEADBAND, CAMERA_HFOV_DEG);
}

void setEnabled(bool on) { s_enabled = on; }

bool takeButtonPress() {
    bool r;
    portENTER_CRITICAL(&s_mux);
    r = (s_btnCount != s_btnTaken);
    s_btnTaken = s_btnCount;
    portEXIT_CRITICAL(&s_mux);
    return r;
}

void onDetection(bool seen, float bearingDeg, float boxRatio, int64_t frameTimeUs) {
    portENTER_CRITICAL(&s_mux);
    s_det.seq++;
    s_det.seen = seen;
    s_det.bearing = bearingDeg;
    s_det.boxRatio = boxRatio;
    s_det.frameUs = frameTimeUs;
    portEXIT_CRITICAL(&s_mux);
}

State state() {
    State s;
    portENTER_CRITICAL(&s_mux);
    s = s_state;
    portEXIT_CRITICAL(&s_mux);
    return s;
}

void printFrameLine() {
    const State s = state();
    char tof[8], vis[8], brg[10];
    if (s.tofMm >= TOF_NONE_MM) snprintf(tof, sizeof tof, "--");
    else snprintf(tof, sizeof tof, "%u", (unsigned)s.tofMm);
    if (s.visMm >= TOF_NONE_MM) snprintf(vis, sizeof vis, "--");
    else snprintf(vis, sizeof vis, "%u", (unsigned)s.visMm);
    if (s.seen) snprintf(brg, sizeof brg, "%+.1f", s.bearingDeg);
    else snprintf(brg, sizeof brg, "--");
    Serial.printf("[Robot] %s brg=%s yawT=%+.1f yaw=%+.1f%s | tof=%s vis=%s ratio=%.2f | v=%.0f/%.0f tgt=%.0f/%.0f pwm=%d/%d | enc=%ld/%ld | ctl=%.1fms | btn=%d/%lu\n",
                  modeName(s.mode), brg, s.yawTargetDeg, s.yawDeg, s.imuGood ? "" : "!", tof, vis, s.boxRatio,
                  s.vL, s.vR, s.tgtL, s.tgtR, s.pwmL, s.pwmR, (long)s.encL, (long)s.encR,
                  s.ctlDtMaxMs, s.btnLevel, (unsigned long)s.btnPresses);
    // Bản gọn cho BLE (~75 ký tự/frame): chế độ, góc, yaw, tof, vis, tốc độ đo/đích, pwm, chu kỳ điều khiển lớn nhất
    BleLog::log("R %s b=%s y=%+.1f%s t=%s vis=%s v=%.0f/%.0f g=%.0f/%.0f p=%d/%d c=%.1f",
                modeName(s.mode), brg, s.yawDeg, s.imuGood ? "" : "!", tof, vis,
                s.vL, s.vR, s.tgtL, s.tgtR, s.pwmL, s.pwmR, s.ctlDtMaxMs);
}

}  // namespace Robot