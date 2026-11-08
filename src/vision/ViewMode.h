#pragma once
#include "CameraModule.h"
#include "RobotConfig.h"   // ROBOT_VIEW_MODE

// ==== CẤU HÌNH WIFI CHO CHẾ ĐỘ XEM ẢNH (chỉ dùng khi gõ 'v') ====
// Thứ tự: thử nối vào WiFi nhà (STA) trong VIEW_STA_TIMEOUT_MS; nếu không nối
// được (hoặc VIEW_WIFI_SSID để trống) thì ESP32 tự phát WiFi riêng (AP), khi đó
// nối laptop vào VIEW_AP_SSID rồi mở http://192.168.4.1
#define VIEW_WIFI_SSID      ""            // tên WiFi nhà/điện thoại; "" = luôn phát WiFi riêng
#define VIEW_WIFI_PASSWORD  "88889999"
#define VIEW_STA_TIMEOUT_MS 10000
#define VIEW_AP_SSID        "ESP32S3-CAM"
#define VIEW_AP_PASSWORD    "12345678"    // tối thiểu 8 ký tự

// Bật/tắt web server + WiFi của Espressif (app_httpd.cpp).
void startCameraServer();
void stopCameraServer();

namespace ViewMode {

// true khi đang ở chế độ VIEW (vòng nhận diện phải tạm dừng).
bool active();

// Đọc lệnh 1 ký tự từ Serial:
//   v = vào chế độ xem ảnh     d = về chế độ nhận diện
//   p = in thông số camera     m = in RAM còn trống     ? = trợ giúp
// Cùng hàm này xử lý nút G0 (BOOT): bấm = đổi qua lại v <-> d.
void handleSerial(CameraModule& camera);

// Lệnh 't' (hoặc nơi khác) yêu cầu chạy self-test; main.cpp lấy yêu cầu đúng một lần bằng takeSelfTestRequest().
void requestSelfTest();
bool takeSelfTestRequest();

}  // namespace ViewMode
