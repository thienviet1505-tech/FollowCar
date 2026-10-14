#include "Robot.h"
#include <Arduino.h>
#include <esp_timer.h>
#include "RobotConfig.h"
#include "RobotMath.h"
#include "Sensors.h"
#include "Drive.h"

using namespace Cfg;

namespace {

using Robot::Mode;

struct Det {
    uint32_t seq = 0;
    bool seen = false;
    float bearing = 0;
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
        case Mode::VIEW: return "VIEW";
        case Mode::LOST: return "LOST";
        case Mode::TRACK: return "TRACK";
        default: return "BLOCK";
    }
}

void controlTask(void*) {
    Sensors::Reading rd;
    RM::WheelPid pidL, pidR;
    RM::YawHistory hist;

    Mode mode = Mode::CALIB;
    uint32_t modeSince = millis();
    uint32_t nextCalibMs = 0;
    bool coasted = false;
    bool warnedNoImu = false;

    uint32_t lastDetSeq = 0;
    uint32_t lastSeenMs = 0;
    int seenStreak = 0;
    float yawTarget = 0;
    bool dispSeen = false;
    float dispBearing = 0;

    float vL = 0, vR = 0, tgtL = 0, tgtR = 0;
    int32_t ePrevL = 0, ePrevR = 0;
    int pwmL = 0, pwmR = 0;
    float winMax = 0, dtShown = 0;
    int winTicks = 0;

    // nút G0: chỉ tính khi đã thấy nhả ra ít nhất 1 lần (tránh lúc bật nguồn đang giữ nút)
    bool btnArmed = false, btnFired = false;
    uint32_t btnLowSince = 0;
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    int btnRaw = digitalRead(BUTTON_PIN), btnRawPrev = btnRaw;
    uint32_t btnPresses = 0;

    Sensors::update(0.0f, rd);
    ePrevL = rd.encL;
    ePrevR = rd.encR;

    TickType_t last = xTaskGetTickCount();
    int64_t tPrev = esp_timer_get_time();

    auto setMode = [&](Mode m) {
        if (m == mode) return;
        Serial.printf("[Robot] %s -> %s\n", modeName(mode), modeName(m));
        mode = m;
        modeSince = millis();
        coasted = false;
        pidL.reset();
        pidR.reset();
        tgtL = tgtR = 0;
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
        if (dt > 0.0f) {
            const float rawL = RM::countsPerSecToMmPerSec((float)(rd.encL - ePrevL) / dt);
            const float rawR = RM::countsPerSecToMmPerSec((float)(rd.encR - ePrevR) / dt);
            vL += SPEED_FILTER_ALPHA * (rawL - vL);
            vR += SPEED_FILTER_ALPHA * (rawR - vR);
        }
        ePrevL = rd.encL;
        ePrevR = rd.encR;

        // ---- nút G0 ----
        btnRaw = digitalRead(BUTTON_PIN);
        const bool btnLow = (btnRaw == LOW);
        if (btnRaw != btnRawPrev) {     // chẩn đoán: in ngay mọi lần chân đổi mức, chưa qua chống dội
            Serial.println(btnLow ? "[Robot] G0 xuong" : "[Robot] G0 len");
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
                s_enabled = false;     // dừng ngay; loop() sẽ đổi chế độ rồi bật lại khi cần
                portENTER_CRITICAL(&s_mux);
                s_btnCount++;
                portEXIT_CRITICAL(&s_mux);
                Serial.println("[Robot] Nut G0 -> dung xe");
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
                yawTarget = hist.at(det.frameUs, rd.yawDeg) + det.bearing;   // yaw lúc chụp + góc lệch
            } else {
                seenStreak = 0;
            }
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
                    Sensors::update(0.0f, rd);
                    ePrevL = rd.encL;
                    ePrevR = rd.encR;
                    vL = vR = 0;
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
            if (seenStreak >= REACQUIRE_FRAMES && (nowMs - lastSeenMs) <= LOST_TIMEOUT_MS && rd.imuGood) setMode(Mode::TRACK);
        } else if (mode == Mode::TRACK || mode == Mode::BLOCKED) {
            if (!rd.imuGood) {
                Serial.println("[Robot] IMU loi -> dung");
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

        // ---- đầu ra ----
        if (mode == Mode::TRACK || mode == Mode::BLOCKED) {
            RM::FollowCmd c = RM::followLaw(yawTarget - rd.yawDeg, rd.tofMm, tofFresh);
            if (!tofFresh || mode == Mode::BLOCKED) c.v = fminf(c.v, 0.0f);   // không có ToF tươi / quá gần: cấm tiến
            RM::wheelTargets(c, tgtL, tgtR);
            pidL.target = tgtL;
            pidR.target = tgtR;
            float fl, fr;
            const bool bl = pidL.step(vL, dtc, fl);
            const bool br = pidR.step(vR, dtc, fr);
            pwmL = bl ? 0 : (int)lroundf(fl);
            pwmR = br ? 0 : (int)lroundf(fr);
            Drive::setPwm(pwmL, pwmR);
        } else {
            tgtL = tgtR = 0;
            pwmL = pwmR = 0;
            if (mode == Mode::VIEW && !coasted && (nowMs - modeSince) >= (uint32_t)BRAKE_HOLD_MS) {
                Drive::coast();         // thả trôi để lăn tay đo encoder
                coasted = true;
            } else if (!coasted) {
                Drive::setPwm(0, 0);    // phanh
            }
        }

        // ---- đăng trạng thái ----
        Robot::State s;
        s.mode = mode;
        s.yawDeg = rd.yawDeg; s.rateDps = rd.rateDps; s.yawTargetDeg = yawTarget;
        s.imuGood = rd.imuGood;
        s.seen = dispSeen; s.bearingDeg = dispBearing;
        s.tofMm = rd.tofMm; s.tofAgeMs = rd.tofAgeMs;
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
    Drive::begin();                  // phanh ngay
    Sensors::begin();
    if (!Sensors::tofPresent()) {
        Serial.println("[Robot] CANH BAO: khong co ToF -> xe chi quay, khong tien/lui");
    }
    xTaskCreatePinnedToCore(controlTask, "robotctl", CTRL_TASK_STACK, nullptr, CTRL_TASK_PRIO, nullptr, CTRL_CORE);
    Serial.printf("[Robot] Task dieu khien %d Hz, lo %d, uu tien %d | lop tien muc tieu=%d | giu %u mm\n",
                  1000 / CTRL_PERIOD_MS, CTRL_CORE, CTRL_TASK_PRIO, TARGET_CLASS, (unsigned)FOLLOW_DISTANCE_MM);
    Serial.printf("[Robot] GEAR_RATIO=%.1f V_AT_PWM_MAX=%.0f PWM_DEADBAND=%.0f HFOV=%.0f (uoc luong - xem huong dan trong RobotConfig.h)\n",
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

void onDetection(bool seen, float bearingDeg, int64_t frameTimeUs) {
    portENTER_CRITICAL(&s_mux);
    s_det.seq++;
    s_det.seen = seen;
    s_det.bearing = bearingDeg;
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
    char tof[8], brg[10];
    if (s.tofMm >= TOF_NONE_MM) snprintf(tof, sizeof tof, "--");
    else snprintf(tof, sizeof tof, "%u", (unsigned)s.tofMm);
    if (s.seen) snprintf(brg, sizeof brg, "%+.1f", s.bearingDeg);
    else snprintf(brg, sizeof brg, "--");
    Serial.printf("[Robot] %s brg=%s yawT=%+.1f yaw=%+.1f%s | tof=%s | v=%.0f/%.0f tgt=%.0f/%.0f pwm=%d/%d | enc=%ld/%ld | ctl=%.1fms | btn=%d/%lu\n",
                  modeName(s.mode), brg, s.yawTargetDeg, s.yawDeg, s.imuGood ? "" : "!", tof, s.vL, s.vR, s.tgtL, s.tgtR,
                  s.pwmL, s.pwmR, (long)s.encL, (long)s.encR, s.ctlDtMaxMs, s.btnLevel, (unsigned long)s.btnPresses);
}

}  // namespace Robot