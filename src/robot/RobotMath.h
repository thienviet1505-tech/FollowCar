#pragma once
// Phần toán thuần (không đụng phần cứng)
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

// Xoay vector ngang (x, y) từ hệ tọa độ MPU6050 (lệch 45 độ) về hệ tọa độ của xe (x = tiến, y = trái)
inline void imuHorizontalToRobot(float imuX, float imuY, float& robotX, float& robotY) {
    const float rad = Cfg::MPU_YAW_MOUNT_DEG / RAD2DEG;
    const float c = cosf(rad), s = sinf(rad);
    robotX = imuX * c - imuY * s;
    robotY = imuX * s + imuY * c;
}

// Tốc độ yaw = gyro . up. up = vector gia tốc lúc đứng yên đã chuẩn hóa (hướng lên trên).
// Vì MPU6050 nằm ngang nên trục up vẫn thẳng đứng, phép chiếu này bất biến với góc xoay ngang 45 độ.
inline float yawRateDps(const float gyroDps[3], const float up[3]) {
    return gyroDps[0] * up[0] + gyroDps[1] * up[1] + gyroDps[2] * up[2];
}

// Góc lệch của tiền so với hướng nhìn của camera (độ, + = tiền ở bên TRÁI).
inline float bearingDeg(float cx, float imgW) {
    const float r = (cx - imgW * 0.5f) / (imgW * 0.5f);
    return -(float)Cfg::BEARING_SIGN * RAD2DEG * atanf(r * tanf(Cfg::CAMERA_HFOV_DEG * 0.5f / RAD2DEG));
}

// Ước lượng khoảng cách tới tờ tiền (mm) dựa trên tỉ lệ kích thước Bounding Box (boxRatio = max(w, h) / imgW)
inline uint16_t estimateVisualDistMm(float boxRatio) {
    if (boxRatio <= 0.02f) return Cfg::TOF_NONE_MM;
    const float dist = (float)Cfg::FOLLOW_DISTANCE_MM * (Cfg::BOX_RATIO_AT_FOLLOW_DIST / boxRatio);
    return (uint16_t)clampf(dist, 50.0f, (float)Cfg::TOF_MAX_MM);
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

// PI vận tốc 1 bánh.
struct WheelPid {
    float target = 0, ramp = 0, integ = 0;

    void reset() { target = ramp = integ = 0; }

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
    float at(int64_t tUs, float fallback) const {
        if (count == 0) return fallback;
        const int newest = (head - 1 + N) % N;
        if (tUs >= t[newest]) return y[newest];
        const int oldest = (head - count + N) % N;
        if (tUs <= t[oldest]) return y[oldest];
        for (int i = 1; i < count; i++) {
            const int b = (head - 1 - (i - 1) + N) % N;
            const int a = (b - 1 + N) % N;
            if (tUs >= t[a] && tUs <= t[b]) {
                const float f = (float)(tUs - t[a]) / (float)(t[b] - t[a] > 0 ? t[b] - t[a] : 1);
                return y[a] + f * (y[b] - y[a]);
            }
        }
        return y[newest];
    }
};

// ---- Luật bám theo (kết hợp cả khoảng cách ToF và khoảng cách thị giác từ Bounding Box) ----
struct FollowCmd {
    float v = 0;   // vận tốc tiến (mm/s), + = tiến
    float w = 0;   // tốc độ quay (do/s), + = quay trái
};

inline FollowCmd followLaw(float yawErr, uint16_t tofMm, bool tofFresh, uint16_t visDistMm) {
    FollowCmd c;
    const float a = fabsf(yawErr);
    const bool tofValid = tofFresh && (tofMm < Cfg::TOF_NONE_MM);
    const bool visValid = (visDistMm < Cfg::TOF_NONE_MM);

    // Kết hợp khoảng cách: lấy khoảng cách gần hơn giữa VL53L0X và ước lượng từ kích thước hộp Camera.
    // Nhờ đó khi giơ tiền trên cao (ToF bắn trượt ra xa), visDistMm vẫn giúp xe dừng đúng ở 15cm!
    uint16_t effDistMm = Cfg::TOF_NONE_MM;
    if (tofValid && visValid) {
        effDistMm = (tofMm < visDistMm) ? tofMm : visDistMm;
    } else if (visValid) {
        effDistMm = visDistMm;
    } else if (tofValid) {
        effDistMm = tofMm;
    }
    const bool hasDist = (effDistMm < Cfg::TOF_NONE_MM);

    if (a <= Cfg::ROTATE_ONLY_DEG) {
        if (hasDist) {
            const float e = (float)effDistMm - (float)Cfg::FOLLOW_DISTANCE_MM;
            if (e > Cfg::FOLLOW_DEADBAND_MM) {
                // Còn xa hơn 15cm + deadband -> tiến lại gần
                c.v = clampf(Cfg::KP_DIST * (e - Cfg::FOLLOW_DEADBAND_MM),
                             0.0f, Cfg::V_FWD_MAX_MM_S);
            } else if (Cfg::ALLOW_REVERSE_WHEN_CLOSE && e < -Cfg::FOLLOW_DEADBAND_MM) {
                // Chỉ lùi nếu bật cờ ALLOW_REVERSE_WHEN_CLOSE
                c.v = clampf(Cfg::KP_DIST_BACK * (e + Cfg::FOLLOW_DEADBAND_MM),
                             -Cfg::V_BACK_MAX_MM_S, 0.0f);
            } else {
                // Đã vào vùng dừng: dừng tiến!
                c.v = 0.0f;
            }
        } else {
            // Trường hợp hiếm khi không có cả ToF lẫn kích thước box: tiến thăm dò chậm
            c.v = Cfg::V_APPROACH_MM_S;
        }
        // Góc lệch lớn: giảm tốc độ tiến tuyến tính (1 tại TOF_USE_BEARING_DEG -> 0 tại ROTATE_ONLY_DEG), không có bước nhảy
        if (c.v > 0.0f && a > Cfg::TOF_USE_BEARING_DEG) {
            const float k = (Cfg::ROTATE_ONLY_DEG - a) / (Cfg::ROTATE_ONLY_DEG - Cfg::TOF_USE_BEARING_DEG);
            c.v = fminf(c.v, Cfg::V_FWD_MAX_MM_S * k);
        }
    }

    // Tốc độ quay liên tục theo sai lệch hướng: bằng 0 tại HEAD_TOL_DEG rồi tăng dần (không có bước nhảy -> không giật)
    float w = 0.0f;
    if (a > Cfg::HEAD_TOL_DEG) {
        w = sgnf(yawErr) * clampf(Cfg::HEAD_KP * (a - Cfg::HEAD_TOL_DEG), 0.0f, Cfg::MAX_TURN_DPS);
    }
    c.w = w;
    return c;
}

// (v, w) -> vận tốc đích 2 bánh
static_assert(Cfg::SPEED_DIVISOR >= 1.0f, "SPEED_DIVISOR phai >= 1");
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
    vL /= Cfg::SPEED_DIVISOR;   // giảm tốc độ để test (SPEED_DIVISOR = 1 -> không đổi)
    vR /= Cfg::SPEED_DIVISOR;
}

}  // namespace RM