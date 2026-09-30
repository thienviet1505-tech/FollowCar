#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H

// ===================
// Chọn model camera/board
// ===================
// Chỉ bật MỘT dòng. Chân GPIO của từng model nằm trong camera_pins.h
// (cùng cách tổ chức với ví dụ CameraWebServer của Espressif).
//
// Board ESP32-S3-N16R8 CAM của bạn đang chạy tốt với bộ chân này (trùng khớp
// với sơ đồ chân ESP32-S3-EYE). Nếu sau này đổi board có chân khác: thêm 1 khối
// #elif defined(CAMERA_MODEL_...) vào camera_pins.h rồi đổi dòng #define dưới.
#define CAMERA_MODEL_ESP32S3_EYE  // Has PSRAM
//#define CAMERA_MODEL_XIAO_ESP32S3 // Has PSRAM

#include "camera_pins.h"

#endif  // BOARD_CONFIG_H
