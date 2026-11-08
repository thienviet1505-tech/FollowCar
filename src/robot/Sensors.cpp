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
Adafruit_VL53L0X tofDev;

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

// ---- Trạng thái 2 bánh, dùng cho ZUPT (Zero-Velocity Update: khóa trôi gyro khi 2 bánh đứng yên) ----
struct Wheels {
    int32_t prevEncL = 0, prevEncR = 0;
    int stationaryTicks = 0;   // số chu kỳ liên tiếp cả 2 bánh không quay (dừng đếm ở STATIONARY_TICKS_MAX)
} g_wheels;

// ---- IMU (MPU6050) ----
struct Imu {
    bool ok = false;           // thấy cảm biến
    bool good = false;         // dữ liệu hiện tại dùng được
    float up[3] = {0, 0, 1};   // hướng "lên" (đơn vị) đo lúc hiệu chỉnh
    float bias[3] = {0, 0, 0}; // độ/s
    float yaw = 0, rate = 0;
    int badTicks = 0;          // số mẫu |accel| bất thường liên tiếp
    // Chống treo (sau sụt áp / nhiễu I2C)
    float prevSample[6] = {0, 0, 0, 0, 0, 0};   // accel xyz + gyro xyz của lần đọc trước
    int frozenTicks = 0;                        // số lần đọc liên tiếp trả đúng y hệt mẫu cũ
    uint32_t badSinceMs = 0;                    // 0 = IMU đang tốt
    uint32_t lastReinitMs = 0;
} g_imu;

// ---- ToF (VL53L0X) ----
constexpr int kMedianN = 3;    // trung vị 3 mẫu: 1 mẫu nhiễu không còn làm xe dừng/lùi (median3 cố định cỡ này)
struct Tof {
    bool ok = false;
    uint16_t mm = TOF_NONE_MM;
    uint32_t stampMs = 0;      // thời điểm có mẫu hợp lệ gần nhất
    uint16_t hist[kMedianN] = {TOF_NONE_MM, TOF_NONE_MM, TOF_NONE_MM};
    int histN = 0, histIdx = 0;
    uint32_t lastReinitMs = 0;
} g_tof;

// Thanh ghi MPU6050
constexpr uint8_t kRegPwrMgmt1 = 0x6B;
constexpr uint8_t kPwrMgmt1WakePllX = 0x01;   // thức, clock = PLL gyro X

void imuConfigure() {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
}

bool imuStart() {
    const bool ok = mpu.begin(MPU_I2C_ADDR, &Wire);
    if (ok) imuConfigure();
    return ok;
}

bool tofStart() {
    const bool ok = tofDev.begin(VL53L0X_I2C_ADDR, false, &Wire);
    if (ok) tofDev.startRangeContinuous(TOF_PERIOD_MS);
    return ok;
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
    Wire.setClock(I2C_CLOCK_HZ);
}

// MPU6050 bị tụt nguồn sẽ về chế độ ngủ (PWR_MGMT_1 = 0x40) và quên cấu hình: đánh thức lại + cấu hình lại.
// Giữ nguyên bias/up đã hiệu chỉnh lúc khởi động (đang chạy thì không thể hiệu chỉnh lại).
void imuReinit() {
    i2cRecover();
    Wire.beginTransmission(MPU_I2C_ADDR);
    Wire.write(kRegPwrMgmt1);
    Wire.write(kPwrMgmt1WakePllX);
    Wire.endTransmission();
    delay(10);
    imuConfigure();
    // Không xóa frozenTicks: nếu cảm biến vẫn đóng băng thì phải tiếp tục báo xấu (xóa đi sẽ làm imuGood nhấp nháy
    // thành "tốt" 0.3 s với dữ liệu cũ sau mỗi lần khởi tạo lại); cảm biến sống lại thì mẫu mới khác mẫu cũ và bộ đếm tự về 0.
}

// VL53L0X mất nguồn sẽ quên chế độ đo liên tục: khởi tạo lại y như lúc boot (mất ~100 ms nên chỉ gọi khi xe đứng yên).
void tofReinit() {
    g_tof.ok = tofStart();
    g_tof.mm = TOF_NONE_MM;
    g_tof.histN = 0;
    g_tof.stampMs = millis();
}

uint16_t median3(uint16_t a, uint16_t b, uint16_t c) {
    const uint16_t lo = a < b ? a : b, hi = a < b ? b : a;
    return c < lo ? lo : (c > hi ? hi : c);
}

// ---- Các bước của Sensors::update ----

// Đọc encoder (đã nhân dấu), cập nhật bộ đếm "2 bánh đứng yên".
void readEncoders(int32_t& encL, int32_t& encR) {
    encL = ENC_SIGN_L * s_encL;
    encR = ENC_SIGN_R * s_encR;
    const bool wheelsMoved = (encL != g_wheels.prevEncL) || (encR != g_wheels.prevEncR);
    g_wheels.prevEncL = encL;
    g_wheels.prevEncR = encR;

    if (wheelsMoved) {
        g_wheels.stationaryTicks = 0;
    } else if (g_wheels.stationaryTicks < STATIONARY_TICKS_MAX) {
        g_wheels.stationaryTicks++;
    }
}

// IMU đóng băng: MPU6050 ngủ / đọc I2C lỗi thì thư viện giữ nguyên mẫu cũ -> 6 số y hệt nhau liên tiếp (cảm biến thật luôn có nhiễu).
bool imuFrozen(const sensors_event_t& a, const sensors_event_t& g) {
    const float cur[6] = {a.acceleration.x, a.acceleration.y, a.acceleration.z, g.gyro.x, g.gyro.y, g.gyro.z};
    bool same = true;
    for (int k = 0; k < 6; k++) {
        if (cur[k] != g_imu.prevSample[k]) same = false;
        g_imu.prevSample[k] = cur[k];
    }
    g_imu.frozenTicks = same ? g_imu.frozenTicks + 1 : 0;
    return IMU_WATCHDOG && g_imu.frozenTicks >= IMU_FROZEN_TICKS;
}

// Tích phân tốc độ góc quanh trục "lên" thành yaw (có ZUPT).
void integrateYaw(const sensors_event_t& g, float dt) {
    const float rawG[3] = {g.gyro.x * RM::RAD2DEG, g.gyro.y * RM::RAD2DEG, g.gyro.z * RM::RAD2DEG};
    const float gd[3] = {rawG[0] - g_imu.bias[0], rawG[1] - g_imu.bias[1], rawG[2] - g_imu.bias[2]};
    const float rawRate = RM::yawRateDps(gd, g_imu.up);

    // ZUPT: Nếu 2 bánh đứng yên đủ lâu và tốc độ góc nhỏ (chỉ là nhiễu/trôi nhiệt),
    // khóa tích phân góc yaw và cập nhật mượt lại bias để triệt tiêu trôi góc.
    if (GYRO_ZUPT && g_wheels.stationaryTicks >= ZUPT_HOLD_TICKS && fabsf(rawRate) < ZUPT_MAX_RATE_DPS) {
        for (int k = 0; k < 3; k++) {
            g_imu.bias[k] += GYRO_BIAS_ADAPT_ALPHA * (rawG[k] - g_imu.bias[k]);
        }
        g_imu.rate = 0.0f;
    } else {
        g_imu.rate = rawRate;
        g_imu.yaw += g_imu.rate * dt;
    }
}

void readImu(float dt) {
    if (!g_imu.ok) return;
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    const bool frozen = imuFrozen(a, g);
    const float am = sqrtf(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y +
                           a.acceleration.z * a.acceleration.z);
    if (frozen) {
        g_imu.good = false;
    } else if (am < IMU_ACCEL_MIN_MS2 || am > IMU_ACCEL_MAX_MS2) {
        if (++g_imu.badTicks >= IMU_BAD_TICKS) g_imu.good = false;
    } else {
        g_imu.badTicks = 0;
        g_imu.good = true;
        integrateYaw(g, dt);
    }
}

// Bộ canh IMU. (1) Không thấy MPU6050 lúc boot (nguồn pin chưa ổn định / nhiễu I2C): thử lại định kỳ. Trước đây chỉ thử 1 lần
// lúc boot rồi xe đứng ở CALIB cho tới khi reset => "lúc đi lúc không" tùy lần bật nguồn.
// (2) Xấu liên tục quá IMU_REINIT_MS thì khởi tạo lại (Robot đã tự dừng xe khi imuGood = false, và tự chạy lại khi tốt).
void imuWatchdog() {
    if (!IMU_WATCHDOG) return;
    const uint32_t nowMs = millis();
    if (!g_imu.ok) {
        if (nowMs - g_imu.lastReinitMs >= IMU_REINIT_RETRY_MS) {
            g_imu.lastReinitMs = nowMs;
            i2cRecover();
            g_imu.ok = imuStart();
            Serial.printf("[Sensors] Thu lai MPU6050: %s\n", g_imu.ok ? "OK" : "van KHONG THAY");
            BleLog::log("SENSOR thu lai MPU6050: %s", g_imu.ok ? "OK" : "van khong thay");
        }
    } else if (g_imu.good) {
        g_imu.badSinceMs = 0;
    } else {
        if (g_imu.badSinceMs == 0) g_imu.badSinceMs = nowMs;
        if (nowMs - g_imu.badSinceMs >= IMU_REINIT_MS && nowMs - g_imu.lastReinitMs >= IMU_REINIT_RETRY_MS) {
            g_imu.lastReinitMs = nowMs;
            Serial.println("[Sensors] IMU loi/dong bang -> khoi tao lai I2C + MPU6050");
            BleLog::log("SENSOR IMU loi/dong bang -> khoi tao lai");
            imuReinit();
        }
    }
}

void readTof() {
    if (!g_tof.ok || !tofDev.isRangeComplete()) return;
    const uint16_t r = tofDev.readRangeResult();
    if (tofDev.Status != VL53L0X_ERROR_NONE) return;
    const uint16_t raw = (r == 0xffff || r >= TOF_MAX_MM) ? TOF_NONE_MM : r;
    const uint16_t v = (raw == TOF_NONE_MM) ? raw : (uint16_t)max(0, (int)raw - TOF_BIAS_MM);   // không tràn số không dấu khi raw < bias
    g_tof.hist[g_tof.histIdx] = v;
    g_tof.histIdx = (g_tof.histIdx + 1) % kMedianN;
    if (g_tof.histN < kMedianN) g_tof.histN++;
    g_tof.mm = (TOF_MEDIAN_FILTER && g_tof.histN >= kMedianN) ? median3(g_tof.hist[0], g_tof.hist[1], g_tof.hist[2]) : v;
    g_tof.stampMs = millis();
}

// Bộ canh ToF: không thấy lúc boot, hoặc lâu không có mẫu mới (mất nguồn), và xe đang đứng yên thì khởi tạo lại.
void tofWatchdog() {
    if (!TOF_WATCHDOG || g_wheels.stationaryTicks < ZUPT_HOLD_TICKS) return;
    const uint32_t nowMs = millis();
    if ((!g_tof.ok || nowMs - g_tof.stampMs > TOF_REINIT_MS) && nowMs - g_tof.lastReinitMs >= TOF_REINIT_RETRY_MS) {
        g_tof.lastReinitMs = nowMs;
        tofReinit();
        Serial.printf("[Sensors] Khoi tao lai VL53L0X: %s\n", g_tof.ok ? "OK" : "van KHONG THAY");
        BleLog::log("SENSOR khoi tao lai ToF: %s", g_tof.ok ? "OK" : "van khong thay");
    }
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
    Wire.setClock(I2C_CLOCK_HZ);

    g_imu.ok = imuStart();
    g_tof.ok = tofStart();
    g_tof.stampMs = g_tof.lastReinitMs = millis();   // để bộ canh ToF không khởi tạo lại ngay lúc boot

    Serial.printf("[Sensors] MPU6050 %s | VL53L0X %s\n", g_imu.ok ? "OK" : "KHONG THAY",
                  g_tof.ok ? "OK" : "KHONG THAY");
    if (!g_imu.ok || !g_tof.ok) {
        Serial.println("[Sensors] Kiem tra day SDA(G1)/SCL(G2)/nguon.");
    }
}

bool imuPresent() { return g_imu.ok; }
bool tofPresent() { return g_tof.ok; }

bool calibrateGyro() {
    if (!g_imu.ok) return false;
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
    if (range > GYRO_CAL_MAX_RANGE_DPS || am < GYRO_CAL_ACCEL_MIN_MS2 || am > GYRO_CAL_ACCEL_MAX_MS2) {
        Serial.printf("[Gyro] HIEU CHINH THAT BAI: robot dang rung/di chuyen? (dao dong=%.1f do/s, |a|=%.1f m/s2).\n",
                      range, am);
        BleLog::log("GYRO calib that bai: dao dong=%.1f |a|=%.1f (giu xe dung yen)", range, am);
        return false;
    }
    for (int k = 0; k < 3; k++) {
        g_imu.bias[k] = sg[k] / n;
        g_imu.up[k] = (sa[k] / n) / am;
    }
    g_imu.yaw = 0;
    g_wheels.prevEncL = ENC_SIGN_L * s_encL;
    g_wheels.prevEncR = ENC_SIGN_R * s_encR;
    g_wheels.stationaryTicks = 0;
    Serial.printf("[Gyro] bias=(%.2f,%.2f,%.2f) do/s | up=(%.2f,%.2f,%.2f) | dao dong=%.2f do/s\n",
                  g_imu.bias[0], g_imu.bias[1], g_imu.bias[2], g_imu.up[0], g_imu.up[1], g_imu.up[2], range);
    BleLog::log("GYRO calib OK bias=%.2f,%.2f,%.2f", g_imu.bias[0], g_imu.bias[1], g_imu.bias[2]);
    return true;
}

void update(float dt, Reading& out) {
    readEncoders(out.encL, out.encR);
    readImu(dt);
    imuWatchdog();
    readTof();
    tofWatchdog();

    out.yawDeg = g_imu.yaw;
    out.rateDps = g_imu.rate;
    out.imuGood = g_imu.good;
    out.tofMm = g_tof.mm;
    out.tofAgeMs = g_tof.ok ? (millis() - g_tof.stampMs) : 0xFFFFFFFFu;
}

}  // namespace Sensors
