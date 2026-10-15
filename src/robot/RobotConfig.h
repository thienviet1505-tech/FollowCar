#pragma once
// =============================================================================
// ROBOT CONFIG - TẤT CẢ tham số cần chỉnh / hiệu chỉnh nằm ở đây.
// (Ngoại lệ: thông số camera - chỉnh bằng chế độ xem ảnh + lệnh 'p' trong CameraModule.cpp;
//  WiFi của chế độ xem ảnh - trong ViewMode.h.)
// Sửa xong chỉ cần nạp lại firmware.
// =============================================================================
#include <stdint.h>

namespace Cfg {

// ============================== BÁM THEO =====================================
constexpr int TARGET_CLASS = 2;                        // lớp tiền cần bám: 0=100k 1=10k 2=50k 3=5k
constexpr float DETECT_CONF_THRESHOLD = 0.5f;          // ngưỡng nhận diện

constexpr uint16_t FOLLOW_DISTANCE_MM = 300;           // khoảng cách giữ với tờ tiền (20 cm)
constexpr float FOLLOW_DEADBAND_MM = 15.0f;            // sai lệch nhỏ hơn mức này thì đứng yên (chống dao động tới lui)
constexpr float KP_DIST = 2.0f;                        // vận tốc (mm/s) = KP_DIST * (sai lệch khoảng cách mm)
constexpr float V_FWD_MAX_MM_S = 150.0f;               // tốc độ tiến tối đa khi bám
constexpr float V_BACK_MAX_MM_S = 50.0f;              // tốc độ lùi tối đa (ToF không nhìn thấy phía sau!)
constexpr float V_APPROACH_MM_S = 50.0f;               // tiến chậm cố định khi không dùng được ToF để đo khoảng cách

constexpr float CAMERA_HFOV_DEG = 65.0f;               // góc nhìn ngang của camera - ƯỚC LƯỢNG, cần đo lại (xem hướng dẫn cuối file)
constexpr int BEARING_SIGN = +1;                       // đổi thành -1 nếu tiền ở bên phải ảnh mà xe lại quay trái

constexpr float ROTATE_ONLY_DEG = 50.0f;               // lệch hơn mức này: chỉ quay tại chỗ, không tiến/lùi
constexpr float TOF_USE_BEARING_DEG = 15.0f;           // chỉ tin ToF là khoảng cách tới tiền khi lệch hướng dưới mức này
constexpr float HEAD_KP = 6.0f;                        // omega (do/s) = HEAD_KP * sai_lech_huong (do)
constexpr float HEAD_TOL_DEG = 3.0f;                   // lệch dưới mức này coi như đã thẳng hướng
constexpr float MAX_TURN_DPS = 180.0f;                 // tốc độ quay tối đa
constexpr float MIN_TURN_DPS = 50.0f;                  // quay tại chỗ chậm hơn mức này bánh không nhúc nhích nổi

constexpr uint32_t LOST_TIMEOUT_MS = 500;              // không thấy tiền quá lâu -> LOST (dừng, phanh)
constexpr int REACQUIRE_FRAMES = 2;                    // cần thấy tiền liên tiếp bao nhiêu frame mới bám lại sau LOST

// ============================== AN TOÀN ToF ==================================
constexpr uint16_t TOF_STOP_MM = 80;                   // đang tiến mà gần hơn mức này -> BLOCKED (dừng tiến)
constexpr uint16_t TOF_RESUME_MM = 100;                // xa hơn mức này mới hết BLOCKED
constexpr uint32_t TOF_STALE_MS = 500;                 // ToF không cập nhật quá lâu thì cấm tiến
constexpr int TOF_PERIOD_MS = 50;                      // chu kỳ đo liên tục của VL53L0X (20 Hz)
constexpr uint16_t TOF_NONE_MM = 9999;                 // quy ước: không thấy vật / đo lỗi
constexpr uint16_t TOF_MAX_MM = 2000;                  // xa hơn mức này coi như không thấy

// ============================== NÚT BẤM ======================================
constexpr int BUTTON_PIN = 0;                          // G0 (BOOT): bấm đổi qua lại chế độ 'd' (nhận diện+bám) và 'v' (xem ảnh)
constexpr int BUTTON_DEBOUNCE_MS = 30;
constexpr uint32_t BRAKE_HOLD_MS = 500;                // sau khi dừng do chuyển chế độ: phanh ngần này ms rồi thả trôi (để lăn tay đo encoder)

// ============================== CHÂN GPIO (khớp sơ đồ phương án B) ===========
constexpr int SDA_PIN = 1, SCL_PIN = 2;                // I2C: MPU6050 (0x68) + VL53L0X (0x29)
constexpr int STBY = 45;                               // cần trở ~2k kéo xuống GND (chân strapping)
constexpr int PWMA = 39, AIN1 = 38, AIN2 = 21;   // bánh TRÁI thật (TB6612 kênh B)
constexpr int PWMB = 42, BIN1 = 41, BIN2 = 40;   // bánh PHẢI thật (TB6612 kênh A)
constexpr int ENC_AL = 46, ENC_BL = 3;           // encoder bánh trái thật
constexpr int ENC_AR = 47, ENC_BR = 14;          // encoder bánh phải thật

// ============================== CHIỀU QUAY ===================================
// Kiểm tra: kê xe lên, bấm G0 (vào chế độ xem ảnh -> bánh thả trôi), quay tay từng bánh về phía TRƯỚC
// và đọc enc= trong dòng [Robot] (in mỗi 1 giây): phải DƯƠNG. Âm -> đổi ENC_SIGN của bánh đó.
// Khi bám theo, nếu bánh nào quay lùi so với lệnh (xe quay sai chiều/lùi khi cần tiến): đổi MOTOR_SIGN.
constexpr int MOTOR_SIGN_L = -1, MOTOR_SIGN_R = -1;
constexpr int ENC_SIGN_L = +1,  ENC_SIGN_R = -1;

// ============================== PWM (LEDC) ===================================
// Camera dùng LEDC kênh 0 / timer 0 cho XCLK -> motor dùng kênh 2, 3 (không đụng).
constexpr int PWM_FREQ_HZ = 20000, PWM_BITS = 8;
constexpr int LEDC_CH_L = 2, LEDC_CH_R = 3;
constexpr int PWM_MAX = 150;                           // trần PWM (0..255)

// ============================== CƠ KHÍ (ước lượng - hiệu chỉnh sau) ==========
constexpr float WHEEL_DIAMETER_MM = 40.0f;             // đường kính bánh
constexpr float TRACK_MM = 150.0f;                     // khoảng cách 2 bánh (tâm-tâm)
// GEAR_RATIO: bấm G0 (bánh thả trôi), lăn tay 1 bánh đúng 1 vòng, đọc enc= trong dòng [Robot]
//   -> GEAR_RATIO = |số đếm| / ENC_PULSES_PER_MOTOR_REV / 4.   (21.3 chỉ là ước lượng; encoder tự về 0 mỗi lần khởi động lại)
constexpr float GEAR_RATIO = 21.3f;
constexpr float ENC_PULSES_PER_MOTOR_REV = 11.0f;      // mỗi kênh, ở trục motor
constexpr float COUNTS_PER_WHEEL_REV = ENC_PULSES_PER_MOTOR_REV * 4.0f * GEAR_RATIO;   // giải mã 4x

// ============================== CÔNG THỨC VẬN TỐC ============================
//   v[mm/s] = (xung/s) / COUNTS_PER_WHEEL_REV * PI * WHEEL_DIAMETER_MM
//   omega[do/s] = (vPhai - vTrai) / TRACK_MM * 180/PI        (quay sang trái = dương)
//
// Quan hệ PWM <-> vận tốc dùng làm feed-forward:
//   PWM = PWM_DEADBAND + (PWM_MAX - PWM_DEADBAND) * v / V_AT_PWM_MAX_MM_S
// Hiệu chỉnh từ log: khi xe đang chạy ổn định, bộ PI tự bù nên pwm= trong dòng [Robot] chính là PWM thực cần.
// Lấy 2 cặp (v, pwm) ổn định - 1 cặp lúc tiến chậm (V_APPROACH), 1 cặp lúc tiến nhanh - rồi giải:
//   s = (pwm2 - pwm1) / (v2 - v1)            (độ dốc, PWM trên mm/s)
//   PWM_DEADBAND = pwm1 - s * v1
//   V_AT_PWM_MAX = (PWM_MAX - PWM_DEADBAND) / s
constexpr float V_AT_PWM_MAX_MM_S = 600.0f;            // ước lượng, nên đo lại
constexpr float PWM_DEADBAND = 60.0f;                  // ước lượng, nên đo lại

// ============================== VÒNG ĐIỀU KHIỂN NHANH ========================
constexpr int CTRL_PERIOD_MS = 10;                     // 100 HzDETECT_CONF_THRESHOLD 
constexpr int CTRL_CORE = 0;                           // loop() chạy lõi 1
constexpr int CTRL_TASK_PRIO = 20;                     // cao hơn worker ESP-NN; nếu ctl= trong [Robot] thường > 15ms thì xem lại
constexpr int CTRL_TASK_STACK = 8192;
constexpr float SPEED_FILTER_ALPHA = 0.3f;             // lọc vận tốc đo (0..1, nhỏ = mượt hơn nhưng trễ hơn)
constexpr float SPD_KP = 0.10f;                        // PWM / (mm/s)
constexpr float SPD_KI = 0.60f;                        // PWM / (mm/s) / s
constexpr float SPD_I_MAX = 60.0f;                     // chặn tích phân (đơn vị PWM)
constexpr float ACCEL_MM_S2 = 2000.0f;                 // giới hạn gia tốc bánh (chống trượt)

// ============================== MPU6050 ======================================
constexpr int GYRO_CAL_SAMPLES = 150;                  // 150 mẫu x 10ms = 1.5 s đứng yên lúc khởi động
constexpr float GYRO_CAL_MAX_RANGE_DPS = 3.0f;         // xe bị rung/di chuyển lúc hiệu chỉnh -> báo lỗi và thử lại
constexpr uint32_t GYRO_RETRY_MS = 1000;               // chờ bao lâu giữa các lần thử lại

// =============================================================================
// HƯỚNG DẪN ĐO CAMERA_HFOV_DEG (từ log Serial):
//   Giữ tờ tiền sát MÉP TRÁI ảnh. Xe sẽ tự quay để đưa tiền về giữa; dòng [Robot] in brg= (góc lệch tiền tính bằng độ
//   theo HFOV đang đặt) và yaw=. Ghi lại yaw trước khi xe quay và sau khi quay xong (hiệu = góc thật).
//   HFOV_mới = HFOV_cũ * (góc thật xe đã quay) / |brg| (brg lúc tiền ở mép ảnh).
// =============================================================================

}  // namespace Cfg
