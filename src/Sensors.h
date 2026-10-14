#pragma once
#include <stdint.h>

// MPU6050 (yaw), VL53L0X (khoảng cách), encoder 2 bánh.
// Mọi hàm (trừ begin) chỉ được gọi từ task điều khiển; Wire không dùng ở nơi khác.
namespace Sensors {

struct Reading {
    float yawDeg = 0;        // yaw tích phân từ gyro, + = quay trái
    float rateDps = 0;       // tốc độ yaw
    bool imuGood = false;    // mẫu IMU gần nhất hợp lệ
    uint16_t tofMm = 9999;   // 9999 = không thấy vật / đo lỗi
    uint32_t tofAgeMs = 0;   // tuổi của mẫu ToF
    int32_t encL = 0;        // đã nhân dấu ENC_SIGN, + = bánh quay tiến
    int32_t encR = 0;
};

// Khởi tạo I2C, IMU, ToF, encoder.
void begin();
bool imuPresent();
bool tofPresent();

// Đứng yên ~1.5 s để đo bias gyro + hướng "lên". In kết quả. true nếu ổn.
bool calibrateGyro();

// Gọi mỗi 10 ms.
void update(float dt, Reading& out);

}  // namespace Sensors
