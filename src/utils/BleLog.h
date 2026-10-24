#pragma once
// Xem log qua Bluetooth LE (BLE UART kiểu Nordic UART Service). ESP32-S3 KHÔNG có Bluetooth Classic nên không dùng được SPP.
//
// Cách dùng: điện thoại mở app BLE UART (Serial Bluetooth Terminal -> chọn BLE, nRF Toolbox UART, nRF Connect...),
// quét thấy "FollowRobot", kết nối, bật Notify (đa số app tự bật). Gửi "off" để tắt BLE; "?" để xem lệnh.
//
// Tiết kiệm SRAM: khi TẮT thì NimBLE + task gửi + bộ đệm đều được giải phóng (log() quay về ngay, không tốn gì).
// Bật/tắt: gõ 'b' ở Serial USB (lưu vào NVS, nhớ qua lần khởi động sau), hoặc gửi "off" qua BLE.
// Muốn loại hẳn khỏi firmware (không tốn flash): đặt -DROBOT_BLE_LOG=0 trong platformio.ini.

#ifndef ROBOT_BLE_LOG
#define ROBOT_BLE_LOG 1
#endif

namespace BleLog {

#if ROBOT_BLE_LOG

void begin();                                   // đọc cờ đã lưu và bật nếu cờ = bật. Gọi CUỐI setup().
bool toggle();                                  // đổi bật <-> tắt (có lưu). Trả về trạng thái mong muốn mới.
bool setEnabled(bool on, bool persist = true);  // trả về true nếu đang bật hoặc sẽ bật khi thoát VIEW
bool enabled();                                 // BLE + bộ đệm đang chạy (đang chiếm SRAM)
bool connected();                               // có điện thoại đang kết nối
void pause();                                   // tạm tắt khi vào chế độ VIEW (WiFi); không đổi cờ đã lưu
void resume();                                  // bật lại nếu cờ đang bật
void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));   // không chặn; tắt thì return ngay

#else

inline void begin() {}
inline bool toggle() { return false; }
inline bool setEnabled(bool, bool = true) { return false; }
inline bool enabled() { return false; }
inline bool connected() { return false; }
inline void pause() {}
inline void resume() {}
inline void log(const char*, ...) {}

#endif

}  // namespace BleLog
