#include "Sensors.h"
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_VL53L0X.h>
#include "RobotConfig.h"
#include "RobotMath.h"
#include "BleLog.h"

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

// ---- ZUPT (Zero-Velocity Update: khóa trôi gyro khi 2 bánh đứng yên) ----
int32_t s_prevEncL = 0, s_prevEncR = 0;
int s_stationaryTicks = 0;

// ---- ToF ----
uint16_t s_tofMm = TOF_NONE_MM;
uint32_t s_tofStampMs = 0;
uint16_t s_tofHist[3] = {TOF_NONE_MM, TOF_NONE_MM, TOF_NONE_MM};   // trung vị 3 mẫu: 1 mẫu nhiễu không còn làm xe dừng/lùi
int s_tofHistN = 0, s_tofHistIdx = 0;

// ---- Chống treo (sau sụt áp / nhiễu I2C) ----
float s_prevSample[6] = {0, 0, 0, 0, 0, 0};   // accel xyz + gyro xyz của lần đọc trước
int s_frozenTicks = 0;                        // số lần đọc liên tiếp trả đúng y hệt mẫu cũ
uint32_t s_imuBadSinceMs = 0;                 // 0 = IMU đang tốt
uint32_t s_lastImuReinitMs = 0;
uint32_t s_lastTofReinitMs = 0;

// ---- Encoder (giải mã vuông pha 4x) ----
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

void imuConfigure() {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
}

// Giải phóng bus I2C nếu một slave đang giữ SDA ở mức thấp (9 xung SCL), rồi bật lại Wire.
void i2cRecover() {
    Wire.end();
    pinMode(SDA_PIN, INPUT_PULLUP);
    pinMode(SCL_PIN, OUTPUT);
    for (int i = 0; i < 9; i++) {
        digitalWrite(SCL_PIN, LOW);
        delayMicroseconds(5);
        digitalWrite(SCL_PIN, HIGH);
        delayMicroseconds(5);
    }
    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
}

// MPU6050 bị tụt nguồn sẽ về chế độ ngủ (PWR_MGMT_1 = 0x40) và quên cấu hình: đánh thức lại + cấu hình lại.
// Giữ nguyên bias/up đã hiệu chỉnh lúc khởi động (đang chạy thì không thể hiệu chỉnh lại).
void imuReinit() {
    i2cRecover();
    Wire.beginTransmission(0x68);
    Wire.write((uint8_t)0x6B);   // PWR_MGMT_1
    Wire.write((uint8_t)0x01);   // thức, clock = PLL gyro X
    Wire.endTransmission();
    delay(10);
    imuConfigure();
    // Không xóa s_frozenTicks: nếu cảm biến vẫn đóng băng thì phải tiếp tục báo xấu (xóa đi sẽ làm imuGood nhấp nháy
    // thành "tốt" 0.3 s với dữ liệu cũ sau mỗi lần khởi tạo lại); cảm biến sống lại thì mẫu mới khác mẫu cũ và bộ đếm tự về 0.
}

// VL53L0X mất nguồn sẽ quên chế độ đo liên tục: khởi tạo lại y như lúc boot (mất ~100 ms nên chỉ gọi khi xe đứng yên).
void tofReinit() {
    s_tofOk = tof.begin(VL53L0X_I2C_ADDR, false, &Wire);
    if (s_tofOk) tof.startRangeContinuous(TOF_PERIOD_MS);
    s_tofMm = TOF_NONE_MM;
    s_tofHistN = 0;
    s_tofStampMs = millis();
}

uint16_t median3(uint16_t a, uint16_t b, uint16_t c) {
    const uint16_t lo = a < b ? a : b, hi = a < b ? b : a;
    return c < lo ? lo : (c > hi ? hi : c);
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
    if (s_imuOk) imuConfigure();

    s_tofOk = tof.begin(VL53L0X_I2C_ADDR, false, &Wire);
    if (s_tofOk) tof.startRangeContinuous(TOF_PERIOD_MS);
    s_tofStampMs = s_lastTofReinitMs = millis();   // để bộ canh ToF không khởi tạo lại ngay lúc boot

    Serial.printf("[Sensors] MPU6050 %s | VL53L0X %s\n", s_imuOk ? "OK" : "KHONG THAY",
                  s_tofOk ? "OK" : "KHONG THAY");
    if (!s_imuOk || !s_tofOk) {
        Serial.println("[Sensors] Kiem tra day SDA(G1)/SCL(G2)/nguon.");
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
        Serial.printf("[Gyro] HIEU CHINH THAT BAI: robot dang rung/di chuyen? (dao dong=%.1f do/s, |a|=%.1f m/s2).\n",
                      range, am);
        BleLog::log("GYRO calib that bai: dao dong=%.1f |a|=%.1f (giu xe dung yen)", range, am);
        return false;
    }
    for (int k = 0; k < 3; k++) {
        s_bias[k] = sg[k] / n;
        s_up[k] = (sa[k] / n) / am;
    }
    s_yaw = 0;
    s_prevEncL = ENC_SIGN_L * s_encL;
    s_prevEncR = ENC_SIGN_R * s_encR;
    s_stationaryTicks = 0;
    Serial.printf("[Gyro] bias=(%.2f,%.2f,%.2f) do/s | up=(%.2f,%.2f,%.2f) | dao dong=%.2f do/s\n",
                  s_bias[0], s_bias[1], s_bias[2], s_up[0], s_up[1], s_up[2], range);
    BleLog::log("GYRO calib OK bias=%.2f,%.2f,%.2f", s_bias[0], s_bias[1], s_bias[2]);
    return true;
}

void update(float dt, Reading& out) {
    const int32_t curEncL = ENC_SIGN_L * s_encL;
    const int32_t curEncR = ENC_SIGN_R * s_encR;
    const bool wheelsMoved = (curEncL != s_prevEncL) || (curEncR != s_prevEncR);
    s_prevEncL = curEncL;
    s_prevEncR = curEncR;

    if (wheelsMoved) {
        s_stationaryTicks = 0;
    } else if (s_stationaryTicks < 1000) {
        s_stationaryTicks++;
    }

    if (s_imuOk) {
        sensors_event_t a, g, t;
        mpu.getEvent(&a, &g, &t);
        // IMU đóng băng: MPU6050 ngủ / đọc I2C lỗi thì thư viện giữ nguyên mẫu cũ -> 6 số y hệt nhau liên tiếp (cảm biến thật luôn có nhiễu).
        const float cur[6] = {a.acceleration.x, a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z};
        bool same = true;
        for (int k = 0; k < 6; k++) {
            if (cur[k] != s_prevSample[k]) same = false;
            s_prevSample[k] = cur[k];
        }
        s_frozenTicks = same ? s_frozenTicks + 1 : 0;
        const bool frozen = IMU_WATCHDOG && s_frozenTicks >= IMU_FROZEN_TICKS;

        const float am = sqrtf(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y +
                               a.acceleration.z * a.acceleration.z);
        if (frozen) {
            s_imuGood = false;
        } else if (am < 4.0f || am > 25.0f) {
            if (++s_badImu >= 5) s_imuGood = false;
        } else {
            s_badImu = 0;
            s_imuGood = true;
            const float rawG[3] = {g.gyro.x * RM::RAD2DEG, g.gyro.y * RM::RAD2DEG, g.gyro.z * RM::RAD2DEG};
            const float gd[3] = {rawG[0] - s_bias[0], rawG[1] - s_bias[1], rawG[2] - s_bias[2]};
            const float rawRate = RM::yawRateDps(gd, s_up);

            // ZUPT: Nếu 2 bánh đứng yên đủ lâu và tốc độ góc nhỏ (chỉ là nhiễu/trôi nhiệt),
            // khóa tích phân góc yaw và cập nhật mượt lại bias để triệt tiêu trôi góc.
            if (GYRO_ZUPT && s_stationaryTicks >= ZUPT_HOLD_TICKS && fabsf(rawRate) < ZUPT_MAX_RATE_DPS) {
                for (int k = 0; k < 3; k++) {
                    s_bias[k] += GYRO_BIAS_ADAPT_ALPHA * (rawG[k] - s_bias[k]);
                }
                s_rate = 0.0f;
            } else {
                s_rate = rawRate;
                s_yaw += s_rate * dt;
            }
        }
    }

    // Bộ canh IMU. (1) Không thấy MPU6050 lúc boot (nguồn pin chưa ổn định / nhiễu I2C): thử lại định kỳ. Trước đây chỉ thử 1 lần
    // lúc boot rồi xe đứng ở CALIB cho tới khi reset => "lúc đi lúc không" tùy lần bật nguồn.
    // (2) Xấu liên tục quá IMU_REINIT_MS thì khởi tạo lại (Robot đã tự dừng xe khi imuGood = false, và tự chạy lại khi tốt).
    if (IMU_WATCHDOG) {
        const uint32_t nowMs = millis();
        if (!s_imuOk) {
            if (nowMs - s_lastImuReinitMs >= IMU_REINIT_RETRY_MS) {
                s_lastImuReinitMs = nowMs;
                i2cRecover();
                s_imuOk = mpu.begin(0x68, &Wire);
                if (s_imuOk) imuConfigure();
                Serial.printf("[Sensors] Thu lai MPU6050: %s\n", s_imuOk ? "OK" : "van KHONG THAY");
                BleLog::log("SENSOR thu lai MPU6050: %s", s_imuOk ? "OK" : "van khong thay");
            }
        } else if (s_imuGood) {
            s_imuBadSinceMs = 0;
        } else {
            if (s_imuBadSinceMs == 0) s_imuBadSinceMs = nowMs;
            if (nowMs - s_imuBadSinceMs >= IMU_REINIT_MS && nowMs - s_lastImuReinitMs >= IMU_REINIT_RETRY_MS) {
                s_lastImuReinitMs = nowMs;
                Serial.println("[Sensors] IMU loi/dong bang -> khoi tao lai I2C + MPU6050");
                BleLog::log("SENSOR IMU loi/dong bang -> khoi tao lai");
                imuReinit();
            }
        }
    }

    if (s_tofOk && tof.isRangeComplete()) {
        const uint16_t r = tof.readRangeResult();
        if (tof.Status == VL53L0X_ERROR_NONE) {
            const uint16_t v = (r == 0xffff || r >= TOF_MAX_MM) ? TOF_NONE_MM : r;
            s_tofHist[s_tofHistIdx] = v;
            s_tofHistIdx = (s_tofHistIdx + 1) % 3;
            if (s_tofHistN < 3) s_tofHistN++;
            s_tofMm = (TOF_MEDIAN_FILTER && s_tofHistN >= 3) ? median3(s_tofHist[0], s_tofHist[1], s_tofHist[2]) : v;
            s_tofStampMs = millis();
        }
    }

    // Bộ canh ToF: không thấy lúc boot, hoặc lâu không có mẫu mới (mất nguồn), và xe đang đứng yên thì khởi tạo lại.
    if (TOF_WATCHDOG && s_stationaryTicks >= ZUPT_HOLD_TICKS) {
        const uint32_t nowMs = millis();
        if ((!s_tofOk || nowMs - s_tofStampMs > TOF_REINIT_MS) && nowMs - s_lastTofReinitMs >= TOF_REINIT_RETRY_MS) {
            s_lastTofReinitMs = nowMs;
            tofReinit();
            Serial.printf("[Sensors] Khoi tao lai VL53L0X: %s\n", s_tofOk ? "OK" : "van KHONG THAY");
            BleLog::log("SENSOR khoi tao lai ToF: %s", s_tofOk ? "OK" : "van khong thay");
        }
    }

    out.yawDeg = s_yaw;
    out.rateDps = s_rate;
    out.imuGood = s_imuGood;
    out.tofMm = s_tofMm;
    out.tofAgeMs = s_tofOk ? (millis() - s_tofStampMs) : 0xFFFFFFFFu;
    out.encL = curEncL;
    out.encR = curEncR;
}

}  // namespace Sensors