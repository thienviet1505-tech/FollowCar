#pragma once
// Phần toán thuần (không đụng phần cứng) - kiểm tra được trên máy tính.
#include <math.h>
#include <stdint.h>
#include "RobotConfig.h"

namespace RM {

constexpr float PI_F = 3.14159265f;
constexpr float RAD2DEG = 180.0f / PI_F;

inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline float sgnf(float x) { return x < 0.0f ? -1.0f : 1.0f; }

// xung/s -> mm/s
inline float countsPerSecToMmPerSec(float cps) {
    return cps / Cfg::COUNTS_PER_WHEEL_REV * PI_F * Cfg::WHEEL_DIAMETER_MM;
}

// omega (do/s) -> vận tốc cộng thêm cho bánh phải (và trừ đi ở bánh trái). Quay trái (CCW) = dương.
inline float wheelSpeedForTurn(float omegaDegPerS) {
    return omegaDegPerS / RAD2DEG * Cfg::TRACK_MM * 0.5f;
}

// Tốc độ yaw = gyro . up. up = vector gia tốc lúc đứng yên đã chuẩn hóa (hướng lên trên).
// Đúng với mọi cách gắn MPU; yaw dương = quay sang trái (ngược chiều kim đồng hồ nhìn từ trên).
inline float yawRateDps(const float gyroDps[3], const float up[3]) {
    return gyroDps[0] * up[0] + gyroDps[1] * up[1] + gyroDps[2] * up[2];
}

// Góc lệch của tiền so với hướng nhìn của camera (độ, + = tiền ở bên TRÁI). cx: tâm hộp (pixel), imgW: bề rộng ảnh.
inline float bearingDeg(float cx, float imgW) {
    const float r = (cx - imgW * 0.5f) / (imgW * 0.5f);              // -1 (mép trái) .. +1 (mép phải)
    return -(float)Cfg::BEARING_SIGN * RAD2DEG * atanf(r * tanf(Cfg::CAMERA_HFOV_DEG * 0.5f / RAD2DEG));
}

inline float slew(float cur, float target, float maxDelta) {
    if (target > cur + maxDelta) return cur + maxDelta;
    if (target < cur - maxDelta) return cur - maxDelta;
    return target;
}

// Feed-forward: vận tốc (mm/s) -> PWM có dấu
inline float feedForwardPwm(float vMmS) {
    const float a = fabsf(vMmS);
    if (a < 1.0f) return 0.0f;
    const float pwm = Cfg::PWM_DEADBAND +
                      (Cfg::PWM_MAX - Cfg::PWM_DEADBAND) * a / Cfg::V_AT_PWM_MAX_MM_S;
    return sgnf(vMmS) * pwm;
}

// PI vận tốc 1 bánh. Trả về PWM có dấu đã chặn trong [-PWM_MAX, PWM_MAX].
struct WheelPid {
    float target = 0, ramp = 0, integ = 0;

    void reset() { target = ramp = integ = 0; }

    // true nếu nên phanh (đích = 0 và đã dừng đà tăng/giảm tốc)
    bool step(float vMeas, float dt, float& pwmOut) {
        ramp = slew(ramp, target, Cfg::ACCEL_MM_S2 * dt);
        if (target == 0.0f && ramp == 0.0f) {
            integ = 0;
            pwmOut = 0;
            return true;
        }
        const float err = ramp - vMeas;
        integ = clampf(integ + Cfg::SPD_KI * err * dt, -Cfg::SPD_I_MAX, Cfg::SPD_I_MAX);
        pwmOut = clampf(feedForwardPwm(ramp) + Cfg::SPD_KP * err + integ,
                        -(float)Cfg::PWM_MAX, (float)Cfg::PWM_MAX);
        return false;
    }
};

// Lịch sử yaw (vòng 64 mẫu = 0.64 s ở 100 Hz) để tra yaw tại thời điểm chụp ảnh (bù độ trễ nhận diện).
struct YawHistory {
    static constexpr int N = 64;
    int64_t t[N] = {0};
    float y[N] = {0};
    int head = 0, count = 0;

    void clear() { head = 0; count = 0; }
    void push(int64_t tUs, float yaw) {
        t[head] = tUs;
        y[head] = yaw;
        head = (head + 1) % N;
        if (count < N) count++;
    }
    // Nội suy tuyến tính; ngoài khoảng thì lấy mẫu gần nhất. Chưa có mẫu nào -> fallback.
    float at(int64_t tUs, float fallback) const {
        if (count == 0) return fallback;
        const int newest = (head - 1 + N) % N;
        if (tUs >= t[newest]) return y[newest];
        const int oldest = (head - count + N) % N;
        if (tUs <= t[oldest]) return y[oldest];
        for (int i = 1; i < count; i++) {
            const int b = (head - 1 - (i - 1) + N) % N;       // mới hơn
            const int a = (b - 1 + N) % N;                     // cũ hơn
            if (tUs >= t[a] && tUs <= t[b]) {
                const float f = (float)(tUs - t[a]) / (float)(t[b] - t[a] > 0 ? t[b] - t[a] : 1);
                return y[a] + f * (y[b] - y[a]);
            }
        }
        return y[newest];
    }
};

// ---- Luật bám theo ----------------------------------------------------------
struct FollowCmd {
    float v = 0;   // vận tốc tiến (mm/s), + = tiến
    float w = 0;   // tốc độ quay (do/s), + = quay trái
};

// yawErr: yaw đích - yaw hiện tại (độ, + = cần quay trái). tofMm/tofFresh: khoảng cách ToF và độ tươi.
inline FollowCmd followLaw(float yawErr, uint16_t tofMm, bool tofFresh) {
    FollowCmd c;
    const float a = fabsf(yawErr);
    const bool tofSeen = tofFresh && tofMm < Cfg::TOF_NONE_MM;

    if (a <= Cfg::ROTATE_ONLY_DEG) {
        if (a <= Cfg::TOF_USE_BEARING_DEG && tofFresh) {
            if (!tofSeen) {
                c.v = Cfg::V_APPROACH_MM_S;                               // ToF không thấy gì: tiến chậm
            } else {
                const float e = (float)tofMm - (float)Cfg::FOLLOW_DISTANCE_MM;   // + = xa quá
                if (fabsf(e) > Cfg::FOLLOW_DEADBAND_MM) {
                    c.v = clampf(Cfg::KP_DIST * (e - sgnf(e) * Cfg::FOLLOW_DEADBAND_MM),
                                 -Cfg::V_BACK_MAX_MM_S, Cfg::V_FWD_MAX_MM_S);
                }
            }
        } else {
            // lệch hướng vừa phải (hoặc ToF cũ): không tin ToF làm khoảng cách. Tiến chậm, trừ khi ToF cho thấy đã quá gần.
            const bool alreadyClose = tofSeen && (float)tofMm < (float)Cfg::FOLLOW_DISTANCE_MM + Cfg::FOLLOW_DEADBAND_MM;
            c.v = alreadyClose ? 0.0f : Cfg::V_APPROACH_MM_S;
        }
    }

    float w = clampf(Cfg::HEAD_KP * yawErr, -Cfg::MAX_TURN_DPS, Cfg::MAX_TURN_DPS);
    if (a <= Cfg::HEAD_TOL_DEG) w = 0.0f;
    else if (c.v == 0.0f && fabsf(w) < Cfg::MIN_TURN_DPS) w = sgnf(yawErr) * Cfg::MIN_TURN_DPS;
    c.w = w;
    return c;
}

// (v, w) -> vận tốc đích 2 bánh; nếu vượt trần thì thu nhỏ cả hai theo cùng tỉ lệ.
inline void wheelTargets(const FollowCmd& c, float& vL, float& vR) {
    const float d = wheelSpeedForTurn(c.w);
    vL = c.v - d;
    vR = c.v + d;
    const float m = fmaxf(fabsf(vL), fabsf(vR));
    if (m > Cfg::V_AT_PWM_MAX_MM_S) {
        const float k = Cfg::V_AT_PWM_MAX_MM_S / m;
        vL *= k;
        vR *= k;
    }
}

}  // namespace RM
