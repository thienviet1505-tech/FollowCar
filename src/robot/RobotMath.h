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

// xung -> mm quãng đường bánh đi được
inline float countsToMm(float counts) {
    return counts / Cfg::COUNTS_PER_WHEEL_REV * PI_F * Cfg::WHEEL_DIAMETER_MM;
}

// xung/s -> mm/s
inline float countsPerSecToMmPerSec(float cps) {
    return countsToMm(cps);
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
    if (boxRatio <= Cfg::BOX_RATIO_MIN) return Cfg::TOF_NONE_MM;
    const float dist = (float)Cfg::FOLLOW_DISTANCE_MM * (Cfg::BOX_RATIO_AT_FOLLOW_DIST / boxRatio);
    return (uint16_t)clampf(dist, Cfg::VIS_DIST_MIN_MM, (float)Cfg::TOF_MAX_MM);
}

inline float slew(float cur, float target, float maxDelta) {
    if (target > cur + maxDelta) return cur + maxDelta;
    if (target < cur - maxDelta) return cur - maxDelta;
    return target;
}

// Feed-forward: vận tốc (mm/s) -> PWM có dấu.
// Liên tục quanh 0: dưới FF_KNEE_MM_S tăng tuyến tính từ 0 lên mức tại điểm gãy (bản cũ nhảy thẳng 0 -> PWM_DEADBAND,
// nên khi đảo chiều PWM nhảy +DEADBAND -> -DEADBAND trong 1 chu kỳ).
inline float feedForwardPwm(float vMmS) {
    const float a = fabsf(vMmS);
    if (!Cfg::FF_SMOOTH_ZERO) {   // bản gốc: nhảy thẳng 0 -> PWM_DEADBAND
        if (a < 1.0f) return 0.0f;
        return sgnf(vMmS) * (Cfg::PWM_DEADBAND + (Cfg::PWM_MAX - Cfg::PWM_DEADBAND) * a / Cfg::V_AT_PWM_MAX_MM_S);
    }
    const float atKnee = Cfg::PWM_DEADBAND +
                         (Cfg::PWM_MAX - Cfg::PWM_DEADBAND) * Cfg::FF_KNEE_MM_S / Cfg::V_AT_PWM_MAX_MM_S;
    const float pwm = (a < Cfg::FF_KNEE_MM_S)
        ? atKnee * a / Cfg::FF_KNEE_MM_S
        : Cfg::PWM_DEADBAND + (Cfg::PWM_MAX - Cfg::PWM_DEADBAND) * a / Cfg::V_AT_PWM_MAX_MM_S;
    return sgnf(vMmS) * pwm;
}

// PI vận tốc 1 bánh.
struct WheelPid {
    float target = 0, ramp = 0, integ = 0, pwmPrev = 0;

    void reset() { target = ramp = integ = pwmPrev = 0; }

    // Trả true khi bánh đã dừng hẳn (đích = 0 và ramp đã về 0): khi đó pwmOut = 0 và tích phân được xóa.
    // Ngược lại trả false và pwmOut = PWM có dấu (chưa làm tròn).
    bool step(float vMeas, float dt, float& pwmOut) {
        ramp = slew(ramp, target, Cfg::ACCEL_MM_S2 * dt);
        if (target == 0.0f && ramp == 0.0f) {
            integ = 0;
            pwmPrev = 0;
            pwmOut = 0;
            return true;
        }
        const float err = ramp - vMeas;
        const float integOld = integ;
        integ = clampf(integ + Cfg::SPD_KI * err * dt, -Cfg::SPD_I_MAX, Cfg::SPD_I_MAX);
        const float cmd = clampf(feedForwardPwm(ramp) + Cfg::SPD_KP * err + integ,
                                 -(float)Cfg::PWM_MAX, (float)Cfg::PWM_MAX);
        // Giới hạn tốc độ đổi PWM (giảm dòng đỉnh / sụt áp). Đang bị giới hạn thì không tích thêm tích phân (chống windup).
        if (Cfg::PWM_SLEW_LIMIT) {
            pwmPrev = slew(pwmPrev, cmd, Cfg::PWM_SLEW_PER_S * dt);
            if (pwmPrev != cmd) integ = integOld;
        } else {
            pwmPrev = cmd;
        }
        pwmOut = pwmPrev;
        return false;
    }
};

// Đo vận tốc 2 bánh (mm/s) trên cửa sổ SPEED_WINDOW_TICKS chu kỳ gần nhất (giảm nhiễu lượng tử hóa 1 xung/chu kỳ).
// push() mỗi chu kỳ điều khiển với giá trị encoder (đã nhân dấu) và thời điểm (us).
struct SpeedEstimator {
    static constexpr int W = Cfg::SPEED_WINDOW_TICKS + 1;
    int32_t encL[W] = {0}, encR[W] = {0};
    int64_t t[W] = {0};
    int head = 0, count = 0;
    float vL = 0, vR = 0;   // mm/s, giữ giá trị cũ cho tới khi đủ 2 mẫu

    void reset() { head = 0; count = 0; vL = vR = 0; }

    void push(int64_t tUs, int32_t l, int32_t r) {
        encL[head] = l;
        encR[head] = r;
        t[head] = tUs;
        head = (head + 1) % W;
        if (count < W) count++;
        if (count < 2) return;
        const int newest = (head - 1 + W) % W;
        const int oldest = (head - count + W) % W;
        const float span = (float)(t[newest] - t[oldest]) * 1e-6f;
        if (span > 0.0f) {
            vL = countsPerSecToMmPerSec((float)(encL[newest] - encL[oldest]) / span);
            vR = countsPerSecToMmPerSec((float)(encR[newest] - encR[oldest]) / span);
        }
    }
};

// Lịch sử một đại lượng theo thời gian (vòng 64 mẫu = 0.64 s ở 100 Hz) để tra giá trị tại thời điểm chụp ảnh
// (bù độ trễ nhận diện). Dùng cho yaw (độ) và cho quãng đường (mm).
struct History {
    static constexpr int N = 64;
    int64_t t[N] = {0};
    float v[N] = {0};
    int head = 0, count = 0;

    void clear() { head = 0; count = 0; }
    void push(int64_t tUs, float value) {
        t[head] = tUs;
        v[head] = value;
        head = (head + 1) % N;
        if (count < N) count++;
    }
    // Giá trị tại thời điểm tUs (nội suy tuyến tính); ngoài khoảng đã lưu thì lấy mẫu gần nhất; chưa có mẫu nào thì trả fallback.
    float at(int64_t tUs, float fallback) const {
        if (count == 0) return fallback;
        const int newest = (head - 1 + N) % N;
        if (tUs >= t[newest]) return v[newest];
        const int oldest = (head - count + N) % N;
        if (tUs <= t[oldest]) return v[oldest];
        for (int i = 1; i < count; i++) {
            const int b = (head - 1 - (i - 1) + N) % N;
            const int a = (b - 1 + N) % N;
            if (tUs >= t[a] && tUs <= t[b]) {
                const float f = (float)(tUs - t[a]) / (float)(t[b] - t[a] > 0 ? t[b] - t[a] : 1);
                return v[a] + f * (v[b] - v[a]);
            }
        }
        return v[newest];
    }
};

// ---- Luật bám theo (kết hợp cả khoảng cách ToF và khoảng cách thị giác từ Bounding Box) ----
struct FollowCmd {
    float v = 0;   // vận tốc tiến (mm/s), + = tiến
    float w = 0;   // tốc độ quay (do/s), + = quay trái
};

// holdDist: trạng thái "đã vào vùng dừng" giữ giữa các lần gọi (trễ kiểu Schmitt: dừng tại FOLLOW_DISTANCE_MM,
// chỉ tiến lại khi xa hơn + FOLLOW_RESUME_MM). rateDps: tốc độ góc yaw hiện tại (gyro) để giảm chấn khi quay.
inline FollowCmd followLaw(float yawErr, float rateDps, uint16_t tofMm, bool tofFresh, uint16_t visDistMm,
                           bool& holdDist) {
    FollowCmd c;
    const float a = fabsf(yawErr);
    const bool tofValid = tofFresh && (tofMm < Cfg::TOF_NONE_MM);
    const bool visValid = (visDistMm < Cfg::TOF_NONE_MM);

    // Kết hợp khoảng cách: lấy khoảng cách gần hơn giữa VL53L0X và ước lượng từ kích thước hộp Camera.
    // Nhờ đó khi giơ tiền trên cao (ToF bắn trượt ra xa), visDistMm vẫn giúp xe dừng đúng ở 15cm!
    uint16_t effDistMm = Cfg::TOF_NONE_MM;
    if (tofValid && visValid) {
        effDistMm = (!Cfg::TOF_FUSE_DISTANCE || visDistMm <= tofMm) ? visDistMm : tofMm;   // TOF_FUSE_DISTANCE = false: tin camera
    } else if (visValid) {
        effDistMm = visDistMm;
    } else if (tofValid) {
        effDistMm = tofMm;
    }
    const bool hasDist = (effDistMm < Cfg::TOF_NONE_MM);

    if (a <= Cfg::ROTATE_ONLY_DEG) {
        if (hasDist) {
            const float e = (float)effDistMm - (float)Cfg::FOLLOW_DISTANCE_MM;
            if (Cfg::FOLLOW_HOLD_HYSTERESIS) {
                if (holdDist) {
                    if (e > Cfg::FOLLOW_RESUME_MM) holdDist = false;   // tiền đã đi xa: bám lại
                } else if (e <= 0.0f) {
                    holdDist = true;                                    // tới đúng 15cm: dừng
                }
                if (!holdDist) {
                    c.v = clampf(Cfg::KP_DIST * e, Cfg::V_FWD_MIN_MM_S, Cfg::V_FWD_MAX_MM_S);
                } else if (Cfg::ALLOW_REVERSE_WHEN_CLOSE && e < -Cfg::FOLLOW_REVERSE_MM) {
                    // Chỉ lùi nếu bật cờ ALLOW_REVERSE_WHEN_CLOSE
                    c.v = clampf(Cfg::KP_DIST_BACK * (e + Cfg::FOLLOW_REVERSE_MM), -Cfg::V_BACK_MAX_MM_S, 0.0f);
                } else {
                    c.v = 0.0f;
                }
            } else if (e > Cfg::FOLLOW_DEADBAND_MM) {
                // Còn xa hơn 15cm + deadband -> tiến lại gần
                c.v = clampf(Cfg::KP_DIST * (e - Cfg::FOLLOW_DEADBAND_MM), 0.0f, Cfg::V_FWD_MAX_MM_S);
            } else if (Cfg::ALLOW_REVERSE_WHEN_CLOSE && e < -Cfg::FOLLOW_DEADBAND_MM) {
                c.v = clampf(Cfg::KP_DIST_BACK * (e + Cfg::FOLLOW_DEADBAND_MM), -Cfg::V_BACK_MAX_MM_S, 0.0f);
            } else {
                c.v = 0.0f;   // đã vào vùng dừng
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

    // Tốc độ quay theo sai lệch hướng, trừ đi một phần tốc độ góc đang quay về phía đích (giảm chấn).
    // Liên tục tại HEAD_TOL_DEG; trong vùng HEAD_TOL_DEG thì đứng yên.
    float w = 0.0f;
    if (a > Cfg::HEAD_TOL_DEG) {
        const float s = sgnf(yawErr);
        const float toward = s * rateDps;
        w = s * clampf(Cfg::HEAD_KP * (a - Cfg::HEAD_TOL_DEG) - Cfg::HEAD_KD * toward, 0.0f, Cfg::MAX_TURN_DPS);
    }
    c.w = w;
    return c;
}

// (v, w) -> vận tốc đích 2 bánh, nhân SPEED_SCALE (< 1 chậm đi, > 1 nhanh lên).
static_assert(Cfg::SPEED_SCALE >= 0.25f && Cfg::SPEED_SCALE <= 2.0f, "SPEED_SCALE phai trong [0.25, 2.0]");
inline void wheelTargets(const FollowCmd& c, float& vL, float& vR) {
    const float d = wheelSpeedForTurn(c.w);
    vL = (c.v - d) * Cfg::SPEED_SCALE;
    vR = (c.v + d) * Cfg::SPEED_SCALE;
    // Trần vận tốc bánh = V_AT_PWM_MAX_MM_S (ứng với PWM_MAX), giữ nguyên tỉ lệ giữa 2 bánh để không đổi hướng đi.
    // Khi chậm đi (SPEED_SCALE < 1) trần thu nhỏ theo cùng tỉ lệ (đúng như SPEED_DIVISOR trước đây); nhanh lên thì trần giữ nguyên.
    const float limit = Cfg::V_AT_PWM_MAX_MM_S * fminf(1.0f, Cfg::SPEED_SCALE);
    const float m = fmaxf(fabsf(vL), fabsf(vR));
    if (m > limit) {
        const float k = limit / m;
        vL *= k;
        vR *= k;
    }
}

}  // namespace RM