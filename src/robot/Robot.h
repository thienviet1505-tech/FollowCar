#pragma once
#include <stdint.h>

namespace Robot {

enum class Mode : uint8_t { CALIB, VIEW, LOST, TRACK, BLOCKED };

struct State {
    Mode mode = Mode::CALIB;
    float yawDeg = 0, rateDps = 0, yawTargetDeg = 0;
    bool imuGood = false;
    bool seen = false;             // frame nhận diện gần nhất có thấy tiền không
    float bearingDeg = 0;          // góc lệch của tiền ở frame đó
    uint16_t tofMm = 9999;         // khoảng cách đo bằng VL53L0X (mm)
    uint16_t visMm = 9999;         // khoảng cách ước lượng từ kích thước Bounding Box (mm)
    float boxRatio = 0;            // max(w,h)/bề rộng ảnh đã lọc (0 = chưa có)
    uint32_t tofAgeMs = 0;
    int32_t encL = 0, encR = 0;
    float vL = 0, vR = 0;          // mm/s đo được
    float tgtL = 0, tgtR = 0;      // mm/s đích
    int pwmL = 0, pwmR = 0;
    float ctlDtMaxMs = 0;
    int btnLevel = 1;
    uint32_t btnPresses = 0;
};

void begin();

void setEnabled(bool on);
bool takeButtonPress();

// Gọi mỗi frame nhận diện.
// boxRatio: max(w, h) / origW của tờ tiền (0 nếu không thấy), dùng để ước lượng khoảng cách khi giơ tiền cao.
void onDetection(bool seen, float bearingDeg, float boxRatio, int64_t frameTimeUs);

// Overload 3 tham số để tương thích ngược nếu gọi kiểu cũ
inline void onDetection(bool seen, float bearingDeg, int64_t frameTimeUs) {
    onDetection(seen, bearingDeg, 0.0f, frameTimeUs);
}

State state();
void printFrameLine();

}  // namespace Robot