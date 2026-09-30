#pragma once
#include "esp_camera.h"

// Chân camera nằm trong camera_pins.h, chọn board ở board_config.h.
#include "board_config.h"

class CameraModule {
public:
    bool begin();
    camera_fb_t* capture();
    void release(camera_fb_t* fb);

    // Bỏ n frame (dùng sau khi khởi động / sau khi rời chế độ xem ảnh, vì
    // frame đầu thường cũ hoặc tối do tự động phơi sáng chưa ổn định).
    void discardFrames(int n);

    // In thông số sensor hiện tại dưới dạng dòng code, dán thẳng vào bảng
    // mặc định trong CameraModule.cpp (hàm makeOV2640()/makeOV3660()).
    void printSettings();

private:
    camera_config_t config_{};
    void applySensorDefaults();
};
