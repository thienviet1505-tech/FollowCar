#include "Drive.h"
#include <Arduino.h>
#include "RobotConfig.h"
#include "RobotMath.h"

using namespace Cfg;

namespace {

// LEDC: core 2.x dùng ledcSetup/ledcAttachPin(kênh), core 3.x dùng ledcAttachChannel/ledcWrite(chân).
void pwmAttach(int pin, int ch) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttachChannel(pin, PWM_FREQ_HZ, PWM_BITS, ch);
#else
    ledcSetup(ch, PWM_FREQ_HZ, PWM_BITS);
    ledcAttachPin(pin, ch);
#endif
}

void pwmWrite(int pin, int ch, int duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    (void)ch;
    ledcWrite(pin, duty);
#else
    (void)pin;
    ledcWrite(ch, duty);
#endif
}

void motor(int in1, int in2, int pwmPin, int ch, int signedPwm) {
    if (signedPwm > 0) {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        pwmWrite(pwmPin, ch, signedPwm);
    } else if (signedPwm < 0) {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        pwmWrite(pwmPin, ch, -signedPwm);
    } else {   // phanh: IN1 = IN2 = HIGH
        digitalWrite(in1, HIGH);
        digitalWrite(in2, HIGH);
        pwmWrite(pwmPin, ch, 255);
    }
}

}  // namespace

namespace Drive {

void begin() {
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(BIN1, OUTPUT);
    pinMode(BIN2, OUTPUT);
    pwmAttach(PWMA, LEDC_CH_L);
    pwmAttach(PWMB, LEDC_CH_R);
    setPwm(0, 0);                    // phanh trước khi bật STBY
    pinMode(STBY, OUTPUT);
    digitalWrite(STBY, HIGH);
}

void setPwm(int left, int right) {
    left = (int)RM::clampf((float)left, -(float)PWM_MAX, (float)PWM_MAX) * MOTOR_SIGN_L;
    right = (int)RM::clampf((float)right, -(float)PWM_MAX, (float)PWM_MAX) * MOTOR_SIGN_R;
    motor(AIN1, AIN2, PWMA, LEDC_CH_L, left);
    motor(BIN1, BIN2, PWMB, LEDC_CH_R, right);
}

void coast() {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
    pwmWrite(PWMA, LEDC_CH_L, 0);
    pwmWrite(PWMB, LEDC_CH_R, 0);
}

}  // namespace Drive
