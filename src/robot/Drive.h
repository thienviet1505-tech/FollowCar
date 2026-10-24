#pragma once

namespace Drive {

void begin();
// brakeAtZero = true: PWM = 0 thì phanh ngắn mạch (dừng dứt khoát). false: nhả trôi (đang bám, không hãm khựng).
void setPwm(int pwmL, int pwmR, bool brakeAtZero = true);
void coast();

}  // namespace Drive