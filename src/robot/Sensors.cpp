#include "Sensors.h"
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_VL53L0X.h>
#include "RobotConfig.h"
#include "RobotMath.h"

using namespace Cfg;

namespace {

Adafruit_MPU6050 mpu;
Adafruit_VL53L0X tof;
bool s_imuOk = false, s_tofOk = false;

// ---- IMU ----
float s_up[3] = {0, 0, 1};
float s_bias[3] = {0, 0, 0};   // do/s
float s_yaw = 0;
int s_badImu = 0;
bool s_imuGood = false;
float s_rate = 0;

// ---- ToF ----
uint16_t s_tofMm = TOF_NONE_MM;
uint32_t s_tofStampMs = 0;

// ---- Encoder (giải mã vuông pha 4x, bảng giống robot_test.ino) ----
volatile int32_t s_encL = 0, s_encR = 0;
volatile uint8_t s_stL = 0, s_stR = 0;
const int8_t QEM[16] = {0, 1, -1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, 0};

void IRAM_ATTR isrL() {
    uint8_t s = (digitalRead(ENC_AL) << 1) | digitalRead(ENC_BL);
    s_encL += QEM[(s_stL << 2) | s];
    s_stL = s;
}
void IRAM_ATTR isrR() {
    uint8_t s = (digitalRead(ENC_AR) << 1) | digitalRead(ENC_BR);
    s_encR += QEM[(s_stR << 2) | s];
    s_stR = s;
}

}  // namespace

namespace Sensors {

void begin() {
    // ---- encoder ----
    pinMode(ENC_AL, INPUT_PULLUP);
    pinMode(ENC_BL, INPUT_PULLUP);
    pinMode(ENC_AR, INPUT_PULLUP);
    pinMode(ENC_BR, INPUT_PULLUP);
    s_stL = (digitalRead(ENC_AL) << 1) | digitalRead(ENC_BL);
    s_stR = (digitalRead(ENC_AR) << 1) | digitalRead(ENC_BR);
    attachInterrupt(digitalPinToInterrupt(ENC_AL), isrL, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_BL), isrL, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_AR), isrR, CHANGE);
    attachInterrupt(digitalPinToInterrupt(ENC_BR), isrR, CHANGE);

    // ---- I2C ----
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);

    s_imuOk = mpu.begin(0x68, &Wire);
    if (s_imuOk) {
        mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
        mpu.setGyroRange(MPU6050_RANGE_500_DEG);
        mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
    }

    s_tofOk = tof.begin(VL53L0X_I2C_ADDR, false, &Wire);
    if (s_tofOk) tof.startRangeContinuous(TOF_PERIOD_MS);

    Serial.printf("[Sensors] MPU6050 %s | VL53L0X %s\n", s_imuOk ? "OK" : "KHONG THAY",
                  s_tofOk ? "OK" : "KHONG THAY");
    if (!s_imuOk || !s_tofOk) {
        Serial.println("[Sensors] Kiem tra day SDA(G1)/SCL(G2)/nguon. Neu camera da chiem I2C port 0 thi bao lai de doi sang Wire1.");
    }
}

bool imuPresent() { return s_imuOk; }
bool tofPresent() { return s_tofOk; }

bool calibrateGyro() {
    if (!s_imuOk) return false;
    float sg[3] = {0, 0, 0}, sa[3] = {0, 0, 0};
    float mn[3] = {1e9f, 1e9f, 1e9f}, mx[3] = {-1e9f, -1e9f, -1e9f};
    for (int i = 0; i < GYRO_CAL_SAMPLES; i++) {
        sensors_event_t a, g, t;
        mpu.getEvent(&a, &g, &t);
        const float gd[3] = {g.gyro.x * RM::RAD2DEG, g.gyro.y * RM::RAD2DEG, g.gyro.z * RM::RAD2DEG};
        const float ad[3] = {a.acceleration.x, a.acceleration.y, a.acceleration.z};
        for (int k = 0; k < 3; k++) {
            sg[k] += gd[k];
            sa[k] += ad[k];
            if (gd[k] < mn[k]) mn[k] = gd[k];
            if (gd[k] > mx[k]) mx[k] = gd[k];
        }
        delay(CTRL_PERIOD_MS);
    }
    float range = 0;
    for (int k = 0; k < 3; k++) range = fmaxf(range, mx[k] - mn[k]);
    const float n = (float)GYRO_CAL_SAMPLES;
    const float am = sqrtf(sa[0] * sa[0] + sa[1] * sa[1] + sa[2] * sa[2]) / n;
    if (range > GYRO_CAL_MAX_RANGE_DPS || am < 7.0f || am > 12.5f) {
        Serial.printf("[Gyro] HIEU CHINH THAT BAI: robot dang rung/di chuyen? (dao dong=%.1f do/s, |a|=%.1f m/s2). Giu nguyen gia tri cu.\n",
                      range, am);
        return false;
    }
    for (int k = 0; k < 3; k++) {
        s_bias[k] = sg[k] / n;
        s_up[k] = (sa[k] / n) / am;
    }
    s_yaw = 0;
    Serial.printf("[Gyro] bias=(%.2f,%.2f,%.2f) do/s | up=(%.2f,%.2f,%.2f) | dao dong=%.2f do/s\n",
                  s_bias[0], s_bias[1], s_bias[2], s_up[0], s_up[1], s_up[2], range);
    return true;
}

void update(float dt, Reading& out) {
    if (s_imuOk) {
        sensors_event_t a, g, t;
        mpu.getEvent(&a, &g, &t);
        const float am = sqrtf(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y +
                               a.acceleration.z * a.acceleration.z);
        if (am < 4.0f || am > 25.0f) {          // đọc lỗi thường trả về toàn số 0
            if (++s_badImu >= 5) s_imuGood = false;
        } else {
            s_badImu = 0;
            s_imuGood = true;
            const float gd[3] = {g.gyro.x * RM::RAD2DEG - s_bias[0], g.gyro.y * RM::RAD2DEG - s_bias[1],
                                 g.gyro.z * RM::RAD2DEG - s_bias[2]};
            s_rate = RM::yawRateDps(gd, s_up);
            s_yaw += s_rate * dt;
        }
    }

    if (s_tofOk && tof.isRangeComplete()) {
        const uint16_t r = tof.readRangeResult();
        if (tof.Status == VL53L0X_ERROR_NONE) {   // lỗi I2C -> không làm mới dấu thời gian
            s_tofMm = (r == 0xffff || r >= TOF_MAX_MM) ? TOF_NONE_MM : r;
            s_tofStampMs = millis();
        }
    }

    out.yawDeg = s_yaw;
    out.rateDps = s_rate;
    out.imuGood = s_imuGood;
    out.tofMm = s_tofMm;
    out.tofAgeMs = s_tofOk ? (millis() - s_tofStampMs) : 0xFFFFFFFFu;
    out.encL = ENC_SIGN_L * s_encL;
    out.encR = ENC_SIGN_R * s_encR;
}

}  // namespace Sensors
