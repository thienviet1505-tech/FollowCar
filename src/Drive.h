#pragma once

// Điều khiển 2 động cơ qua TB6612. Chỉ gọi từ task điều khiển (sau begin()).
namespace Drive {

void begin();                       // cấu hình chân + LEDC, bật STBY, phanh cả 2 bánh

// PWM có dấu, -PWM_MAX..PWM_MAX (đã nhân MOTOR_SIGN bên trong). 0 = phanh.
void setPwm(int left, int right);

// Thả trôi cả 2 bánh (IN1 = IN2 = LOW): lăn tay được. Chỉ dùng sau khi đã phanh xong.
void coast();

}  // namespace Drive
