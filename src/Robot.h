#pragma once
#include <stdint.h>

// Tầng điều khiển nhanh: 1 task FreeRTOS (100 Hz, lõi 0) lo IMU, ToF, encoder, PID bánh, nút G0, an toàn
// và máy trạng thái bám theo. loop() (nhận diện ảnh, ~6 FPS) chỉ báo kết quả nhận diện và đọc trạng thái.
namespace Robot {

//  CALIB   : đang hiệu chỉnh gyro (xe phải đứng yên); chưa cho chạy
//  VIEW    : chế độ xem ảnh ('v'), xe dừng
//  LOST    : chế độ nhận diện nhưng chưa thấy tiền (hoặc vừa mất), xe dừng + phanh
//  TRACK   : đang bám tiền
//  BLOCKED : đang bám nhưng ToF báo quá gần, cấm tiến (vẫn quay / lùi được)
enum class Mode : uint8_t { CALIB, VIEW, LOST, TRACK, BLOCKED };

struct State {
    Mode mode = Mode::CALIB;
    float yawDeg = 0, rateDps = 0, yawTargetDeg = 0;
    bool imuGood = false;
    bool seen = false;             // frame nhận diện gần nhất có thấy tiền không
    float bearingDeg = 0;          // góc lệch của tiền ở frame đó
    uint16_t tofMm = 9999;
    uint32_t tofAgeMs = 0;
    int32_t encL = 0, encR = 0;
    float vL = 0, vR = 0;          // mm/s đo được
    float tgtL = 0, tgtR = 0;      // mm/s đích
    int pwmL = 0, pwmR = 0;
    float ctlDtMaxMs = 0;          // chu kỳ dài nhất của vòng điều khiển trong ~2 s gần nhất
    int btnLevel = 1;              // mức thô chân G0 (1 = nhả, 0 = đang bấm)
    uint32_t btnPresses = 0;       // số lần bấm đã được nhận (sau chống dội)
};

void begin();                      // gọi 1 lần trong setup(); gyro tự hiệu chỉnh trong task (xe phải đứng yên)

// true = chế độ nhận diện/bám; false = xem ảnh (xe dừng). Gọi từ ViewMode.
void setEnabled(bool on);
// true 1 lần cho mỗi lần bấm nút G0 (đã chống dội). Lúc bấm, xe đã tự dừng ngay trong task điều khiển.
bool takeButtonPress();

// Gọi mỗi frame nhận diện. frameTimeUs: thời điểm chụp frame (đồng hồ esp_timer_get_time()).
void onDetection(bool seen, float bearingDeg, int64_t frameTimeUs);

State state();
void printFrameLine();             // 1 dòng gọn, in mỗi frame (và mỗi 1 s khi ở chế độ xem ảnh)

}  // namespace Robot