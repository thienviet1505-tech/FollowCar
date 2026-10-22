#include "Drive.h"
#include <Arduino.h>
#include "RobotConfig.h"

using namespace Cfg;

namespace {

void driveOne(int pwm, int sign, int in1, int in2, int ch) {
    const int v = constrain(pwm * sign, -PWM_MAX, PWM_MAX);
    if (v > 0) {
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        ledcWrite(ch, v);
    } else if (v < 0) {
        digitalWrite(in1, LOW);
        digitalWrite(in2, HIGH);
        ledcWrite(ch, -v);
    } else {
        // Phanh ngắn mạch (Short brake) trên TB6612 để xe dừng dứt khoát
        digitalWrite(in1, HIGH);
        digitalWrite(in2, HIGH);
        ledcWrite(ch, 0);
    }
}

}  // namespace

namespace Drive {

void begin() {
    pinMode(STBY, OUTPUT);
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(BIN1, OUTPUT);
    pinMode(BIN2, OUTPUT);

    ledcSetup(LEDC_CH_L, PWM_FREQ_HZ, PWM_BITS);
    ledcAttachPin(PWMA, LEDC_CH_L);

    ledcSetup(LEDC_CH_R, PWM_FREQ_HZ, PWM_BITS);
    ledcAttachPin(PWMB, LEDC_CH_R);

    coast();
}

void setPwm(int pwmL, int pwmR) {
    digitalWrite(STBY, HIGH);
    driveOne(pwmL, MOTOR_SIGN_L, AIN1, AIN2, LEDC_CH_L);
    driveOne(pwmR, MOTOR_SIGN_R, BIN1, BIN2, LEDC_CH_R);
}

void coast() {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
    ledcWrite(LEDC_CH_L, 0);
    ledcWrite(LEDC_CH_R, 0);
    digitalWrite(STBY, LOW);
}

}  // namespace Drive