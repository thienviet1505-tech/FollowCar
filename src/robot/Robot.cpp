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

// ---- Nút G0 (chống dội) ----
// Chỉ có tác dụng khi bật BUTTON_ENABLE và còn chế độ VIEW (ROBOT_VIEW_MODE); tắt thì không đọc GPIO0.
struct Button {
    static constexpr bool kEnabled = BUTTON_ENABLE && (ROBOT_VIEW_MODE != 0);

    int raw = HIGH, rawPrev = HIGH;
    bool armed = false, fired = false;
    uint32_t lowSince = 0;
    uint32_t presses = 0;

    void begin() {
        if (!kEnabled) return;
        pinMode(BUTTON_PIN, INPUT_PULLUP);
        raw = digitalRead(BUTTON_PIN);
        rawPrev = raw;
    }

    // Gọi mỗi chu kỳ. Trả true đúng một lần khi vừa có một lần bấm đủ BUTTON_DEBOUNCE_MS.
    // Phải nhả nút (mức HIGH) rồi mới nhận lần bấm tiếp theo; nút đang bị giữ lúc khởi động không tính.
    bool poll(uint32_t nowMs) {
        if (!kEnabled) return false;
        raw = digitalRead(BUTTON_PIN);
        const bool low = (raw == LOW);
        if (raw != rawPrev) {
            Serial.println(low ? "[Robot] G0 xuong" : "[Robot] G0 len");
            BleLog::log("G0 %s", low ? "xuong" : "len");
            rawPrev = raw;
        }
        if (!low) {
            armed = true;
            fired = false;
            lowSince = 0;
            return false;
        }
        if (!armed || fired) return false;
        if (lowSince == 0) lowSince = nowMs;
        if (nowMs - lowSince < (uint32_t)BUTTON_DEBOUNCE_MS) return false;
        fired = true;
        presses++;
        return true;
    }
};

// ---- Theo dõi mục tiêu từ kết quả nhận diện ----
// Giữ hướng tuyệt đối của tiền (yaw) và khoảng cách ước lượng từ box (có bù quãng đường xe đã đi).
struct TargetTracker {
    uint32_t lastSeq = 0;
    uint32_t lastSeenMs = 0;
    int seenStreak = 0;                    // số frame "thấy" liên tiếp (hụt thì giảm dần), quyết định LOST -> TRACK
    float yawTarget = 0;
    bool hasYawTarget = false;
    float filtBoxRatio = 0;
    float visBaseMm = 0, visBaseOdo = 0;   // khoảng cách ước lượng từ box tại lúc chụp frame + quãng đường lúc đó
    bool hasVis = false;
    bool holdDist = false;                 // đang ở vùng dừng 15cm (trễ Schmitt trong followLaw)
    uint16_t visDistMm = TOF_NONE_MM;
    bool dispSeen = false;                 // kết quả frame gần nhất, chỉ để hiển thị
    float dispBearing = 0;

    // Bỏ mục tiêu hiện tại (không đụng seenStreak/lastSeenMs).
    void forget() {
        hasYawTarget = false;
        hasVis = false;
        holdDist = false;
        filtBoxRatio = 0;
        visDistMm = TOF_NONE_MM;
    }

    // Gọi mỗi chu kỳ: nhận kết quả nhận diện mới (nếu có) và cập nhật khoảng cách hiện tại.
    void update(const Det& det, uint32_t nowMs, float yawNow, float odo,
                const RM::History& yawHist, const RM::History& odoHist) {
        if (det.seq != lastSeq) {
            lastSeq = det.seq;
            dispSeen = det.seen;
            dispBearing = det.bearing;
            if (det.seen) {
                seenStreak++;
                lastSeenMs = nowMs;
                trackYaw(det, yawNow, yawHist);
                trackDistance(det, odo, odoHist);
            } else if (seenStreak > 0) {
                // Giảm dần seenStreak thay vì xóa trắng ngay nếu mới hụt 1 frame
                seenStreak--;
            }
        }
        // Khoảng cách hiện tại = khoảng cách lúc chụp - quãng đường đã đi từ đó (không chạy theo số cũ khi hụt frame)
        if (hasVis) {
            visDistMm = (uint16_t)RM::clampf(visBaseMm - (DIST_DEAD_RECKON ? odo - visBaseOdo : 0.0f),
                                             VIS_DIST_MIN_MM, (float)TOF_MAX_MM);
        }
    }

private:
    // Hướng tuyệt đối của tiền (tại lúc chụp frame), lọc mượt giữa các frame
    void trackYaw(const Det& det, float yawNow, const RM::History& yawHist) {
        const float z = yawHist.at(det.frameUs, yawNow) + det.bearing;
        if (!hasYawTarget) {
            yawTarget = z;
            hasYawTarget = true;
        } else {
            yawTarget += YAW_TARGET_FILTER_ALPHA * (z - yawTarget);
        }
    }

    // Khoảng cách từ box, trộn với giá trị cũ đã trừ quãng đường xe đi được từ lúc đó đến lúc chụp frame
    void trackDistance(const Det& det, float odo, const RM::History& odoHist) {
        const uint16_t meas = RM::estimateVisualDistMm(det.boxRatio);
        if (meas >= TOF_NONE_MM) return;
        const float odoF = DIST_DEAD_RECKON ? odoHist.at(det.frameUs, odo) : 0.0f;
        if (hasVis) {
            const float pred = visBaseMm - (DIST_DEAD_RECKON ? odoF - visBaseOdo : 0.0f);
            visBaseMm = pred + BOX_RATIO_FILTER_ALPHA * ((float)meas - pred);
        } else {
            visBaseMm = (float)meas;
            hasVis = true;
        }
        visBaseMm = RM::clampf(visBaseMm, VIS_DIST_MIN_MM, (float)TOF_MAX_MM);
        visBaseOdo = odoF;
        filtBoxRatio = (float)FOLLOW_DISTANCE_MM * BOX_RATIO_AT_FOLLOW_DIST / visBaseMm;
    }
};

// ---- Vòng điều khiển nhanh (CTRL_PERIOD_MS) ----
// Mỗi chu kỳ: đọc cảm biến -> nút G0 -> nhận diện mới -> máy trạng thái -> PID bánh -> đăng trạng thái.
class ControlLoop {
public:
    void run() {   // không bao giờ trả về
        button_.begin();
        Sensors::update(0.0f, rd_);
        speed_.reset();
        last_ = xTaskGetTickCount();
        tPrev_ = esp_timer_get_time();
        for (;;) {
            vTaskDelayUntil(&last_, pdMS_TO_TICKS(CTRL_PERIOD_MS));
            tick();
        }
    }

private:
    struct Tick {
        int64_t now = 0;       // us
        uint32_t nowMs = 0;
        float dt = 0;          // s, chu kỳ thực tế
        float dtPid = 0;       // s, dt đã kẹp cho PID
        float odo = 0;         // mm, quãng đường trung bình 2 bánh
    };

    void tick() {
        Tick t;
        t.now = esp_timer_get_time();
        t.nowMs = millis();
        t.dt = (float)(t.now - tPrev_) * 1e-6f;
        tPrev_ = t.now;
        t.dtPid = t.dt < CTRL_DT_MAX_PID_S ? t.dt : CTRL_DT_MAX_PID_S;

        trackJitter(t.dt);
        readSensors(t);
        pollButton(t.nowMs);

        Det det;
        portENTER_CRITICAL(&s_mux);
        det = s_det;
        portEXIT_CRITICAL(&s_mux);
        tracker_.update(det, t.nowMs, rd_.yawDeg, t.odo, yawHist_, odoHist_);

        const bool enabled = s_enabled;
        const bool tofFresh = Sensors::tofPresent() && rd_.tofAgeMs <= TOF_STALE_MS;
        updateMode(t.nowMs, enabled, tofFresh);
        drive(t, tofFresh);
        publish();
    }

    // Chu kỳ điều khiển trễ nhất trong cửa sổ CTRL_STATS_TICKS (để báo trong log).
    void trackJitter(float dt) {
        const float dtMs = dt * 1e3f;
        if (dtMs > winMax_) winMax_ = dtMs;
        if (++winTicks_ >= CTRL_STATS_TICKS) {
            dtShown_ = winMax_;
            winMax_ = 0;
            winTicks_ = 0;
        }
    }

    void readSensors(Tick& t) {
        Sensors::update(t.dt < CTRL_DT_MAX_SENSOR_S ? t.dt : CTRL_DT_MAX_SENSOR_S, rd_);
        yawHist_.push(t.now, rd_.yawDeg);
        t.odo = RM::countsToMm(0.5f * (float)(rd_.encL + rd_.encR));   // quay tại chỗ thì 2 bánh triệt tiêu nhau
        odoHist_.push(t.now, t.odo);
        speed_.push(t.now, rd_.encL, rd_.encR);
    }

    void pollButton(uint32_t nowMs) {
        if (!button_.poll(nowMs)) return;
        s_enabled = false;
        portENTER_CRITICAL(&s_mux);
        s_btnCount++;
        portEXIT_CRITICAL(&s_mux);
        Serial.println("[Robot] Nut G0 -> dung xe");
        BleLog::log("G0 -> dung xe");
    }

    void setMode(Mode m) {
        if (m == mode_) return;
        Serial.printf("[Robot] %s -> %s\n", modeName(mode_), modeName(m));
        BleLog::log("MODE %s->%s", modeName(mode_), modeName(m));
        mode_ = m;
        modeSince_ = millis();
        coasted_ = false;
        // Chỉ xóa PID (dừng cứng) khi vào VIEW/CALIB. Với LOST/TRACK/BLOCKED giữ đà để giảm tốc/đổi tốc êm.
        if (m == Mode::VIEW || m == Mode::CALIB) {
            pidL_.reset();
            pidR_.reset();
        }
        tgtL_ = tgtR_ = 0;
        if (m == Mode::LOST || m == Mode::VIEW || m == Mode::CALIB) tracker_.forget();
    }

    // Hiệu chỉnh gyro (xe phải đứng yên). Thất bại thì thử lại sau GYRO_RETRY_MS.
    void calibrateStep(uint32_t nowMs, bool enabled) {
        if (!Sensors::imuPresent()) {
            if (!warnedNoImu_) {
                Serial.println("[Robot] KHONG THAY IMU -> xe chua chay, Sensors se thu lai dinh ky");
                BleLog::log("KHONG THAY IMU -> xe chua chay, dang thu lai");
                warnedNoImu_ = true;
            }
            return;
        }
        if (nowMs < nextCalibMs_) return;
        Serial.println("[Robot] Hieu chinh gyro 1.5 s - giu xe DUNG YEN");
        BleLog::log("CALIB gyro 1.5 s - giu xe dung yen");
        if (Sensors::calibrateGyro()) {
            yawHist_.clear();
            odoHist_.clear();
            Sensors::update(0.0f, rd_);
            speed_.reset();
            tracker_.seenStreak = 0;
            setMode(enabled ? Mode::LOST : Mode::VIEW);
        } else {
            nextCalibMs_ = millis() + GYRO_RETRY_MS;
        }
        // calibrateGyro() chặn ~1.5 s: đặt lại mốc thời gian để chu kỳ kế tiếp không thấy dt khổng lồ
        tPrev_ = esp_timer_get_time();
        last_ = xTaskGetTickCount();
    }

    void updateMode(uint32_t nowMs, bool enabled, bool tofFresh) {
        if (mode_ == Mode::CALIB) {
            calibrateStep(nowMs, enabled);
        } else if (!enabled) {
            setMode(Mode::VIEW);
        } else if (mode_ == Mode::VIEW) {
            tracker_.seenStreak = 0;
            setMode(Mode::LOST);
        }

        if (mode_ == Mode::LOST) {
            if (tracker_.seenStreak >= REACQUIRE_FRAMES && (nowMs - tracker_.lastSeenMs) <= LOST_TIMEOUT_MS && rd_.imuGood) {
                setMode(Mode::TRACK);
            }
        } else if (mode_ == Mode::TRACK || mode_ == Mode::BLOCKED) {
            if (!rd_.imuGood) {
                Serial.println("[Robot] IMU loi -> dung");
                BleLog::log("IMU loi -> dung");
                tracker_.seenStreak = 0;
                setMode(Mode::LOST);
            } else if (nowMs - tracker_.lastSeenMs > LOST_TIMEOUT_MS) {
                tracker_.seenStreak = 0;
                setMode(Mode::LOST);
            } else if (mode_ == Mode::TRACK && TOF_SAFETY_STOP && tofFresh && rd_.tofMm < TOF_STOP_MM) {
                setMode(Mode::BLOCKED);
            } else if (mode_ == Mode::BLOCKED && (!tofFresh || rd_.tofMm >= TOF_RESUME_MM)) {
                setMode(Mode::TRACK);
            }
        }
    }

    // Đặt vận tốc đích tgtL_/tgtR_ cho 2 PID bánh, tính PWM và xuất ra động cơ.
    void runWheelPid(float dtPid) {
        pidL_.target = tgtL_;
        pidR_.target = tgtR_;
        float fl, fr;
        const bool idleL = pidL_.step(speed_.vL, dtPid, fl);
        const bool idleR = pidR_.step(speed_.vR, dtPid, fr);
        pwmL_ = idleL ? 0 : (int)lroundf(fl);
        pwmR_ = idleR ? 0 : (int)lroundf(fr);
        Drive::setPwm(pwmL_, pwmR_, !COAST_AT_ZERO);
    }

    void drive(const Tick& t, bool tofFresh) {
        if (mode_ == Mode::TRACK || mode_ == Mode::BLOCKED) {
            RM::FollowCmd c = RM::followLaw(tracker_.yawTarget - rd_.yawDeg, rd_.rateDps, rd_.tofMm, tofFresh,
                                            tracker_.visDistMm, tracker_.holdDist);
            // Nếu ToF báo có vật cản sát (< TOF_STOP_MM), khóa chiều tiến (nhưng vẫn cho phép xoay hướng theo tiền)
            if (mode_ == Mode::BLOCKED) c.v = fminf(c.v, 0.0f);
            RM::wheelTargets(c, tgtL_, tgtR_);
            runWheelPid(t.dtPid);
        } else if (mode_ == Mode::LOST) {
            // Mất tiền: giảm tốc êm theo ACCEL_MM_S2 (đích = 0)
            tgtL_ = tgtR_ = 0;
            runWheelPid(t.dtPid);
        } else {
            tgtL_ = tgtR_ = 0;
            pwmL_ = pwmR_ = 0;
            if (mode_ == Mode::VIEW && !coasted_ && (t.nowMs - modeSince_) >= (uint32_t)BRAKE_HOLD_MS) {
                Drive::coast();
                coasted_ = true;
            } else if (!coasted_) {
                Drive::setPwm(0, 0);
            }
        }
    }

    void publish() {
        Robot::State s;
        s.mode = mode_;
        s.yawDeg = rd_.yawDeg; s.rateDps = rd_.rateDps; s.yawTargetDeg = tracker_.yawTarget;
        s.imuGood = rd_.imuGood;
        s.seen = tracker_.dispSeen; s.bearingDeg = tracker_.dispBearing;
        s.tofMm = rd_.tofMm; s.visMm = tracker_.visDistMm; s.boxRatio = tracker_.filtBoxRatio; s.tofAgeMs = rd_.tofAgeMs;
        s.encL = rd_.encL; s.encR = rd_.encR;
        s.vL = speed_.vL; s.vR = speed_.vR; s.tgtL = tgtL_; s.tgtR = tgtR_;
        s.pwmL = pwmL_; s.pwmR = pwmR_;
        s.ctlDtMaxMs = dtShown_;
        s.btnLevel = button_.raw;
        s.btnPresses = button_.presses;
        portENTER_CRITICAL(&s_mux);
        s_state = s;
        portEXIT_CRITICAL(&s_mux);
    }

    Sensors::Reading rd_;
    RM::WheelPid pidL_, pidR_;
    RM::SpeedEstimator speed_;
    RM::History yawHist_, odoHist_;     // yaw (độ) và quãng đường (mm) theo thời gian, để bù độ trễ nhận diện
    TargetTracker tracker_;
    Button button_;

    Mode mode_ = Mode::CALIB;
    uint32_t modeSince_ = millis();
    uint32_t nextCalibMs_ = 0;
    bool coasted_ = false;
    bool warnedNoImu_ = false;

    float tgtL_ = 0, tgtR_ = 0;         // mm/s đích
    int pwmL_ = 0, pwmR_ = 0;

    float winMax_ = 0, dtShown_ = 0;
    int winTicks_ = 0;

    TickType_t last_ = 0;
    int64_t tPrev_ = 0;
};

void controlTask(void*) {
    ControlLoop loop;   // nằm trên stack của task, giống các biến cục bộ trước đây
    loop.run();
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
    if (!SERIAL_FRAME_LOG && !BleLog::enabled()) return;   // không ai nhận thì khỏi định dạng chuỗi
    const State s = state();
    char tof[8], vis[8], brg[10];
    if (s.tofMm >= TOF_NONE_MM) snprintf(tof, sizeof tof, "--");
    else snprintf(tof, sizeof tof, "%u", (unsigned)s.tofMm);
    if (s.visMm >= TOF_NONE_MM) snprintf(vis, sizeof vis, "--");
    else snprintf(vis, sizeof vis, "%u", (unsigned)s.visMm);
    if (s.seen) snprintf(brg, sizeof brg, "%+.1f", s.bearingDeg);
    else snprintf(brg, sizeof brg, "--");
    if (SERIAL_FRAME_LOG) {
        Serial.printf("[Robot] %s brg=%s yawT=%+.1f yaw=%+.1f%s | tof=%s vis=%s ratio=%.2f | v=%.0f/%.0f tgt=%.0f/%.0f pwm=%d/%d | enc=%ld/%ld | ctl=%.1fms | btn=%d/%lu\n",
                      modeName(s.mode), brg, s.yawTargetDeg, s.yawDeg, s.imuGood ? "" : "!", tof, vis, s.boxRatio,
                      s.vL, s.vR, s.tgtL, s.tgtR, s.pwmL, s.pwmR, (long)s.encL, (long)s.encR,
                      s.ctlDtMaxMs, s.btnLevel, (unsigned long)s.btnPresses);
    }
    // Bản gọn cho BLE (~75 ký tự/frame): chế độ, góc, yaw, tof, vis, tốc độ đo/đích, pwm, chu kỳ điều khiển lớn nhất
    BleLog::log("R %s b=%s y=%+.1f%s t=%s vis=%s v=%.0f/%.0f g=%.0f/%.0f p=%d/%d c=%.1f",
                modeName(s.mode), brg, s.yawDeg, s.imuGood ? "" : "!", tof, vis,
                s.vL, s.vR, s.tgtL, s.tgtR, s.pwmL, s.pwmR, s.ctlDtMaxMs);
}

}  // namespace Robot