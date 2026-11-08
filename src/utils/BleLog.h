#pragma once
// Xem log qua Bluetooth LE (BLE UART kiểu Nordic UART Service). ESP32-S3 KHÔNG có Bluetooth Classic nên không dùng được SPP.
//
// Bật/tắt bằng MỘT cờ build duy nhất trong RobotConfig.h (đổi xong phải build lại + nạp lại):
//   ROBOT_BLE_LOG = 1   biên dịch BLE và xe tự phát ngay khi boot: điện thoại quét thấy "FollowRobot" là kết nối được
//   ROBOT_BLE_LOG = 0   loại hẳn NimBLE khỏi firmware (không tốn flash/SRAM)
// Điện thoại mở app BLE UART (Serial Bluetooth Terminal -> chọn BLE, nRF Toolbox UART, nRF Connect...), kết nối, bật Notify
// (đa số app tự bật). Bộ đệm giữ ~8 s log gần nhất kể cả khi chưa kết nối, nên vừa kết nối là thấy lý do reset/khởi động.
// Ngắt kết nối thì xe tự phát lại. Riêng chế độ VIEW (WiFi) cần SRAM nội nên BLE tạm tắt rồi tự bật lại khi thoát VIEW.

#include "RobotConfig.h"

namespace BleLog {

#if ROBOT_BLE_LOG

void begin();                                   // bật BLE + task gửi. Gọi CUỐI setup() (sau khi camera/model đã lấy bộ nhớ).
bool enabled();                                 // BLE + bộ đệm đang chạy (đang chiếm SRAM)
void pause();                                   // tạm tắt khi vào chế độ VIEW (WiFi)
void resume();                                  // bật lại khi thoát VIEW
void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));   // không chặn; tắt thì return ngay

#else

inline void begin() {}
inline bool enabled() { return false; }
inline void pause() {}
inline void resume() {}
inline void log(const char*, ...) {}

#endif

}  // namespace BleLog
