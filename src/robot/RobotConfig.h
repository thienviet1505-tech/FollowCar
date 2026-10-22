#pragma once
// =============================================================================
// ROBOT CONFIG - TẤT CẢ tham số cần chỉnh / hiệu chỉnh nằm ở đây.
// =============================================================================
#include <stdint.h>

namespace Cfg {

// ============================== TỐC ĐỘ TEST ==================================
// Chia toàn bộ vận tốc đích 2 bánh (cả tiến lẫn quay) cho hệ số này để test xe cho dễ.
// 1.0 = tốc độ bình thường, 2.0 = chậm 2 lần, 3.0 = chậm 3 lần. Chia quá nhiều (>= ~4) có thể làm xe giật vì dưới PWM_DEADBAND.
constexpr float SPEED_DIVISOR = 1.1f;

// ============================== BÁM THEO =====================================
constexpr int TARGET_CLASS = 2;                        // lớp tiền cần bám: 0=100k 1=10k 2=50k 3=5k

// Ngưỡng kép (Hysteresis) giúp bắt mục tiêu nhạy ở ~6 FPS mà không bị báo giả:
// - DETECT_CONF_THRESHOLD: ngưỡng giải mã cơ bản cho Detector::decode()
// - DETECT_ACQUIRE_THRESH: ngưỡng để bắt đầu nhận tiền khi đang LOST
// - DETECT_HOLD_THRESH   : ngưỡng để duy trì bám khi vừa thấy tiền ở frame trước
constexpr float DETECT_CONF_THRESHOLD = 0.35f;
constexpr float DETECT_ACQUIRE_THRESH = 0.45f;
constexpr float DETECT_HOLD_THRESH    = 0.35f;

constexpr uint16_t FOLLOW_DISTANCE_MM = 150;           // khoảng cách dừng trước tờ tiền: 150 mm (15 cm)
constexpr float FOLLOW_DEADBAND_MM = 15.0f;            // sai lệch nhỏ hơn mức này thì đứng yên
constexpr bool ALLOW_REVERSE_WHEN_CLOSE = true;        // true = lùi khi tiền quá gần; false = chỉ dừng

// Dừng theo kích thước Bounding Box khi giơ tiền cao (VL53L0X không bắn trúng).
// boxRatio = max(w,h) / bề rộng ảnh (tối đa ~1.0). Với 0.85: tiến khi boxRatio < ~0.77 (123 px), dừng trong ~0.77..0.94, lùi khi > ~0.94 (151 px).
// Công thức: dừng/lùi theo khoảng cách ước lượng = FOLLOW_DISTANCE * BOX_RATIO_AT_FOLLOW_DIST / boxRatio; chỉnh lại khi chạy thực tế:
// - Xe dừng khi còn quá xa: TĂNG BOX_RATIO_AT_FOLLOW_DIST.
// - Xe tiến quá sát mới dừng: GIẢM BOX_RATIO_AT_FOLLOW_DIST.
constexpr float BOX_RATIO_AT_FOLLOW_DIST = 0.85f;
constexpr float BOX_RATIO_FILTER_ALPHA   = 0.50f;      // lọc mượt kích thước box giữa các frame

constexpr float KP_DIST = 1.8f;                        // vận tốc (mm/s) = KP_DIST * (sai lệch khoảng cách mm)
constexpr float KP_DIST_BACK = 6.0f;                   // như KP_DIST nhưng cho chiều LÙI (box không to hơn ảnh nên cần hệ số lớn hơn)
constexpr float V_FWD_MAX_MM_S = 140.0f;               // tốc độ tiến tối đa khi bám
constexpr float V_BACK_MAX_MM_S = 50.0f;               // tốc độ lùi tối đa (chỉ dùng khi ALLOW_REVERSE_WHEN_CLOSE = true)
constexpr float V_APPROACH_MM_S = 55.0f;               // tốc độ tiến thăm dò khi chưa rõ khoảng cách

constexpr float CAMERA_HFOV_DEG = 65.0f;               // góc nhìn ngang của camera OV2640
constexpr int BEARING_SIGN = +1;                       // đổi thành -1 nếu tiền ở bên phải ảnh mà xe lại quay trái
constexpr float YAW_TARGET_FILTER_ALPHA = 0.80f;       // làm mượt góc đích yawT giữa các frame (0..1)

constexpr float ROTATE_ONLY_DEG = 45.0f;               // lệch hơn mức này: xoay tại chỗ hướng về tiền trước khi tiến
constexpr float TOF_USE_BEARING_DEG = 20.0f;           // lệch hơn mức này: tốc độ tiến giảm tuyến tính về 0 tại ROTATE_ONLY_DEG
constexpr float HEAD_KP = 10.5f;                        // omega (do/s) = HEAD_KP * (|sai_lech_huong| - HEAD_TOL_DEG), liên tục tại HEAD_TOL_DEG
constexpr float HEAD_TOL_DEG = 4.0f;                   // lệch dưới mức này coi như đã thẳng hướng
constexpr float MAX_TURN_DPS = 160.0f;                 // tốc độ quay tối đa

constexpr uint32_t LOST_TIMEOUT_MS = 1000;             // giữ trạng thái bám tối đa 1000ms (~5-6 frame ở 5.5 FPS)
constexpr int REACQUIRE_FRAMES = 1;                    // với ngưỡng kép (Acquire 0.45), thấy 1 frame chắc chắn là bám ngay

// ============================== AN TOÀN ToF ==================================
constexpr uint16_t TOF_STOP_MM = 120;                  // bất kỳ vật cản nào gần hơn 12cm trước VL53L0X -> dừng tiến ngay
constexpr uint16_t TOF_RESUME_MM = 140;                // xa hơn 14cm mới cho phép tiến lại
constexpr uint32_t TOF_STALE_MS = 500;                 // ToF không cập nhật quá lâu
constexpr int TOF_PERIOD_MS = 50;                      // chu kỳ đo liên tục của VL53L0X (20 Hz)
constexpr uint16_t TOF_NONE_MM = 9999;                 // quy ước: không thấy vật / ngoài tầm
constexpr uint16_t TOF_MAX_MM = 2000;                  // xa hơn mức này coi như không thấy

// ============================== NÚT BẤM ======================================
constexpr int BUTTON_PIN = 0;                          // G0 (BOOT): bấm đổi qua lại chế độ 'd' và 'v'
constexpr int BUTTON_DEBOUNCE_MS = 30;
constexpr uint32_t BRAKE_HOLD_MS = 500;                // phanh 500ms rồi thả trôi khi vào chế độ xem ảnh

// ============================== CHÂN GPIO ====================================
constexpr int SDA_PIN = 1, SCL_PIN = 2;                // I2C: MPU6050 (0x68) + VL53L0X (0x29)
constexpr int STBY = 45;                               // chân STBY của TB6612
constexpr int PWMA = 39, AIN1 = 38, AIN2 = 21;         // bánh TRÁI
constexpr int PWMB = 42, BIN1 = 41, BIN2 = 40;         // bánh PHẢI
constexpr int ENC_AL = 46, ENC_BL = 3;                 // encoder bánh trái
constexpr int ENC_AR = 47, ENC_BR = 14;                // encoder bánh phải

// ============================== CHIỀU QUAY ===================================
constexpr int MOTOR_SIGN_L = -1, MOTOR_SIGN_R = -1;
constexpr int ENC_SIGN_L = +1,  ENC_SIGN_R = -1;

// ============================== PWM (LEDC) ===================================
constexpr int PWM_FREQ_HZ = 20000, PWM_BITS = 8;
constexpr int LEDC_CH_L = 2, LEDC_CH_R = 3;
constexpr int PWM_MAX = 150;

// ============================== CƠ KHÍ (GA25 630 RPM) ========================
constexpr float WHEEL_DIAMETER_MM = 40.0f;             // đường kính bánh ~4cm (40mm)
constexpr float TRACK_MM = 150.0f;                     // khoảng cách 2 bánh ~15cm (150mm)
constexpr float GEAR_RATIO = 9.6f;                     // động cơ GA25 630 RPM có tỉ số truyền ~1:9.6 (hoặc 1:10)
constexpr float ENC_PULSES_PER_MOTOR_REV = 11.0f;      // 11 xung/vòng trục motor mỗi kênh
constexpr float COUNTS_PER_WHEEL_REV = ENC_PULSES_PER_MOTOR_REV * 4.0f * GEAR_RATIO;

constexpr float V_AT_PWM_MAX_MM_S = 600.0f;
constexpr float PWM_DEADBAND = 30.0f;

// ============================== VÒNG ĐIỀU KHIỂN NHANH ========================
constexpr int CTRL_PERIOD_MS = 10;                     // 100 Hz
constexpr int CTRL_CORE = 0;                           // loop() chạy lõi 1
constexpr int CTRL_TASK_PRIO = 20;
constexpr int CTRL_TASK_STACK = 8192;
constexpr int SPEED_WINDOW_TICKS = 5;                   // đo vận tốc bánh trên cửa sổ 5 chu kỳ x 10 ms = 50 ms (1 xung ~ 6 mm/s thay vì ~30 mm/s)
constexpr float SPD_KP = 0.20f;
constexpr float SPD_KI = 2.60f;
constexpr float SPD_I_MAX = 60.0f;
constexpr float ACCEL_MM_S2 = 800.0f;                // gia tốc tối đa của vận tốc đích bánh (thấp = khởi động/dừng êm, đỡ giật)

// ============================== MPU6050 ======================================
constexpr float MPU_YAW_MOUNT_DEG = 45.0f;             // MPU6050 nằm ngang, lệch 45 độ so với phương di chuyển
constexpr int GYRO_CAL_SAMPLES = 150;                  // 150 mẫu x 10ms = 1.5 s đứng yên lúc khởi động
constexpr float GYRO_CAL_MAX_RANGE_DPS = 3.0f;
constexpr uint32_t GYRO_RETRY_MS = 1000;

// Chống trôi góc Yaw khi xe đứng yên (Zero-Velocity Update - ZUPT):
constexpr int ZUPT_HOLD_TICKS = 8;                     // 8 chu kỳ x 10ms = 80ms cả 2 bánh không quay -> kích hoạt ZUPT
constexpr float ZUPT_MAX_RATE_DPS = 2.5f;              // ngưỡng tốc độ góc tối đa coi là nhiễu trôi khi đứng yên
constexpr float GYRO_BIAS_ADAPT_ALPHA = 0.005f;        // tốc độ tự thích nghi bias nhiệt khi xe đứng yên

}  // namespace Cfg