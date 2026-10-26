#pragma once
// =============================================================================
// ROBOT CONFIG - TẤT CẢ tham số cần chỉnh / hiệu chỉnh nằm ở đây.
// =============================================================================
#include <stdint.h>

// =============================================================================
// CỜ BẬT/TẮT TÍNH NĂNG: đổi giá trị -> BUILD LẠI -> NẠP LẠI. Mặc định bật hết (= hành vi hiện tại).
// Lúc boot xe in một dòng "[Cfg] ..." (Serial + BLE) tóm tắt cờ và giờ build để biết đang chạy bản nào.
//   A. Macro (đổi cái được biên dịch/liên kết vào firmware) - ngay bên dưới.
//   B. Cờ hành vi `constexpr bool` - đầu namespace Cfg, ngay sau macro.
//   C. Các cờ cũ DETECT_* / DIST_DEAD_RECKON / FOLLOW_HOLD_HYSTERESIS / COAST_AT_ZERO / ALLOW_REVERSE_WHEN_CLOSE
//      vẫn nằm ở mục BÁM THEO bên dưới.
// =============================================================================

// ---- A. Macro ----
// Log Bluetooth LE (xem src/utils/BleLog.h). 1 = xe tự phát "FollowRobot" ngay khi boot để điện thoại kết nối xem log;
// 0 = loại hẳn NimBLE khỏi firmware (không tốn flash/SRAM). Đây là công tắc DUY NHẤT của BLE (không còn lệnh bật/tắt lúc chạy).
#define ROBOT_BLE_LOG 1
// Chế độ xem ảnh qua WiFi (lệnh 'v' / nút G0). 0 = bỏ WiFi + web server + trang web khỏi firmware (giảm flash/SRAM),
// đồng thời nút G0 không còn tác dụng.
#define ROBOT_VIEW_MODE 1
// Suy luận model (xem ModelInference.cpp). Chỉ đổi khi so sánh tốc độ.
#define MODEL_IN_PSRAM 1            // 1 = model nằm trong PSRAM
#define NN_DUAL_CORE 1              // 1 = ESP-NN chạy 2 lõi
#define CONV_OPT_AUTOTUNE 1         // 1 = tự đo từng conv 3x3 lúc khởi động và chọn cách chạy nhanh nhất (0 = kernel ESP-NN gốc)
#define PROFILE_OPS 0               // 1 = in bảng thời gian từng op sau mỗi lần invoke
#define NN_SCRATCH_PREALLOC 65536   // byte scratch cấp sẵn cho ESP-NN dual-core (0 = tắt); conv 3x3 K=1152 cần 62.560 B

static_assert((ROBOT_BLE_LOG == 0 || ROBOT_BLE_LOG == 1) && (ROBOT_VIEW_MODE == 0 || ROBOT_VIEW_MODE == 1),
              "cac co macro chi nhan 0 hoac 1");

namespace Cfg {

// ---- B. Cờ hành vi (true = bật). false = quay về cách làm của bản gốc cho riêng tính năng đó ----
constexpr bool PWM_SLEW_LIMIT = true;       // giới hạn tốc độ đổi PWM đầu ra (PWM_SLEW_PER_S) + chống windup tích phân
constexpr bool FF_SMOOTH_ZERO = true;       // feed-forward tăng dần từ 0 (FF_KNEE_MM_S); false = nhảy 0 -> PWM_DEADBAND
constexpr bool IMU_WATCHDOG = true;         // MPU6050 đóng băng/lỗi, hoặc KHÔNG THẤY lúc boot -> tự nhả bus I2C + khởi tạo lại (tắt: thiếu IMU lúc boot thì xe đứng cho tới khi reset)
constexpr bool TOF_WATCHDOG = true;         // VL53L0X mất mẫu quá TOF_REINIT_MS hoặc không thấy lúc boot -> tự khởi tạo lại (chỉ khi xe đứng yên)
constexpr bool CAMERA_WATCHDOG = true;      // camera capture lỗi liên tiếp CAMERA_FAIL_REBOOT_COUNT lần (camera kẹt do nhiễu/sụt áp) -> tự khởi động lại xe
constexpr int CAMERA_FAIL_REBOOT_COUNT = 3;
constexpr bool TOF_MEDIAN_FILTER = true;    // lọc trung vị 3 mẫu ToF (1 mẫu nhiễu không làm xe dừng/lùi); false = phản ứng nhanh hơn 50-100 ms
constexpr bool TOF_FUSE_DISTANCE = true;    // khoảng cách bám = min(ToF, ước lượng từ box); false = ưu tiên camera, ToF chỉ dùng khi camera không có số
constexpr bool TOF_SAFETY_STOP = true;      // ToF < TOF_STOP_MM thì khóa chiều tiến (BLOCKED). false = MẤT AN TOÀN, chỉ để thử nghiệm
constexpr bool BUTTON_ENABLE = true;        // nút G0 (BOOT) đổi VIEW/DETECT. false = không đọc GPIO0 (loại nghi vấn nhiễu làm xe dừng). Cần ROBOT_VIEW_MODE = 1
constexpr bool GYRO_ZUPT = true;            // khóa trôi gyro khi 2 bánh đứng yên (ZUPT_*)
constexpr bool REACQUIRE_GRACE = true;      // sau khi buông mục tiêu, REACQUIRE_GRACE_FRAMES frame kế bắt lại bằng ngưỡng DETECT_HOLD_THRESH
constexpr bool SERIAL_FRAME_LOG = true;     // in ~500 ký tự/frame ra Serial (tốn ~40 ms/frame ở 115200, kể cả khi không ai đọc). false = chỉ in sự kiện/lỗi; log BLE gọn vẫn chạy
constexpr bool BOOT_SELFTEST = true;        // chạy self-test model lúc boot (lệnh 't' vẫn chạy thủ công được). false = boot nhanh hơn, nhưng mất bước làm nóng + kiểm tra model

// ============================== TỐC ĐỘ =======================================
// Nhân toàn bộ vận tốc đích 2 bánh (cả tiến lẫn quay) với hệ số này. Cho phép [0.25, 2.0].
// 1.0 = bình thường, 0.5 = chậm một nửa, 1.5 = nhanh hơn 50%. (Thay cho SPEED_DIVISOR cũ: SPEED_SCALE = 1 / SPEED_DIVISOR.)
// - Chậm đi (< 1): quá nhỏ (< ~0.25) thì vận tốc đích rơi xuống dưới PWM_DEADBAND, xe giật.
// - Nhanh lên (> 1): vẫn bị chặn ở V_AT_PWM_MAX_MM_S (ứng với PWM_MAX). Luật bám/quay được tính TRƯỚC khi nhân hệ số nên độ lợi
//   vòng điều khiển (KP_DIST, HEAD_KP) cũng tăng theo hệ số: nếu xe lắc khi quay hoặc khi dừng thì giảm hệ số hoặc HEAD_KP.
constexpr float SPEED_SCALE = 1.0f;

// ============================== BÁM THEO =====================================
constexpr int TARGET_CLASS = 2;                        // lớp tiền cần bám: 0=100k 1=10k 2=50k 3=5k

// Ngưỡng kép (Hysteresis) giúp bắt mục tiêu nhạy ở ~6 FPS mà không bị báo giả:
// - DETECT_CONF_THRESHOLD: ngưỡng giải mã cơ bản cho Detector::decode() (phải <= DETECT_GATE_THRESH)
// - DETECT_ACQUIRE_THRESH: ngưỡng để bắt đầu nhận tiền khi đang LOST
// - DETECT_HOLD_THRESH   : ngưỡng để duy trì bám khi vừa thấy tiền ở frame trước (box ở chỗ khác)
// - DETECT_GATE_THRESH   : ngưỡng cho box nằm đúng chỗ dự kiến của tiền đang bám (xem GATE_*)
constexpr float DETECT_CONF_THRESHOLD = 0.25f;
constexpr float DETECT_ACQUIRE_THRESH = 0.45f;
constexpr float DETECT_HOLD_THRESH    = 0.35f;
constexpr float DETECT_GATE_THRESH    = 0.30f;   // ngưỡng cố định trong cổng (khi DETECT_SCORE_SMOOTH = false)
constexpr float DETECT_GATE_MIN_THRESH = 0.25f;  // ngưỡng thấp nhất trong cổng (khi DETECT_SCORE_SMOOTH và mục tiêu đang bám chắc)
static_assert(DETECT_CONF_THRESHOLD <= DETECT_GATE_MIN_THRESH && DETECT_GATE_MIN_THRESH <= DETECT_GATE_THRESH &&
              DETECT_GATE_THRESH <= DETECT_HOLD_THRESH,
              "can DETECT_CONF_THRESHOLD <= DETECT_GATE_MIN_THRESH <= DETECT_GATE_THRESH <= DETECT_HOLD_THRESH");

// ---- Các tính năng thêm vào để giảm khựng / mất tiền: bật-tắt từng cái để kiểm chứng trên xe thật ----
// Tất cả false = hành vi như bản gốc (ngoài các thông số rẽ/ACCEL). Nếu xe xấu đi, tắt lần lượt từng cái để tìm thủ phạm.
constexpr bool DETECT_MULTI_LABEL = true;       // mỗi box được xét theo TỪNG class (không chỉ class điểm cao nhất): hết mất 50k khi 10k nhỉnh hơn
constexpr bool DETECT_GATE = true;              // cổng quanh vị trí dự kiến của tiền: hạ ngưỡng trong cổng, giữ cổng qua frame hụt
constexpr bool DETECT_SCORE_SMOOTH = true;      // (3a, cần DETECT_GATE) điểm EMA của mục tiêu đang bám quyết định ngưỡng trong cổng: bám chắc thì chịu được frame tụt conf
constexpr bool DIST_DEAD_RECKON = true;         // khoảng cách giữa 2 frame = khoảng cách lúc chụp - quãng đường encoder (không chạy theo số cũ)
constexpr bool FOLLOW_HOLD_HYSTERESIS = true;   // dừng đúng 15cm, trễ khi tiến lại; false = deadband +-FOLLOW_DEADBAND_MM như bản gốc
constexpr bool COAST_AT_ZERO = true;            // PWM = 0 thì nhả trôi; false = phanh ngắn mạch như bản gốc
static_assert(!DETECT_SCORE_SMOOTH || DETECT_GATE, "DETECT_SCORE_SMOOTH can DETECT_GATE = true");

// Cổng (gate, chỉ khi DETECT_GATE): box lệch hướng không quá GATE_BEARING_DEG (đã bù góc xe quay giữa 2 frame)
// và cạnh lớn nằm trong [GATE_SIZE_MIN, GATE_SIZE_MAX] x cạnh lớn lần trước thì được nhận ở DETECT_GATE_THRESH;
// trong cổng chọn box ĐIỂM CAO NHẤT. Hụt tối đa GATE_MAX_MISS_FRAMES frame liên tiếp vẫn giữ cổng.
// Làm mượt điểm (DETECT_SCORE_SMOOTH): EMA = SCORE_EMA_ALPHA * điểm frame này + (1 - SCORE_EMA_ALPHA) * EMA cũ (hụt = điểm 0).
// Ngưỡng trong cổng trượt tuyến tính: EMA <= SCORE_STRONG_LOW -> DETECT_HOLD_THRESH; EMA >= SCORE_STRONG_HIGH -> DETECT_GATE_MIN_THRESH.
constexpr float SCORE_EMA_ALPHA = 0.5f;
constexpr float SCORE_STRONG_LOW = 0.30f;
constexpr float SCORE_STRONG_HIGH = 0.50f;
constexpr float GATE_BEARING_DEG = 15.0f;
constexpr float GATE_SIZE_MIN = 0.6f;
constexpr float GATE_SIZE_MAX = 1.6f;
constexpr int GATE_MAX_MISS_FRAMES = 3;                // (cũ 2) hụt tối đa chừng này frame liên tiếp vẫn giữ mục tiêu
// Sau khi buông mục tiêu, trong REACQUIRE_GRACE_FRAMES frame kế (~1.5 s ở 5.5 FPS) bắt lại bằng ngưỡng DETECT_HOLD_THRESH
// thay vì DETECT_ACQUIRE_THRESH (xe chạy giật làm ảnh nhòe, điểm tụt dưới 0.45 nên trước đây xe đứng chờ rất lâu). 0 = tắt.
constexpr int REACQUIRE_GRACE_FRAMES = 8;

constexpr uint16_t FOLLOW_DISTANCE_MM = 150;           // khoảng cách dừng trước tờ tiền: 150 mm (15 cm)
constexpr float FOLLOW_DEADBAND_MM = 15.0f;            // chỉ dùng khi FOLLOW_HOLD_HYSTERESIS = false: sai lệch nhỏ hơn mức này thì đứng yên
// Khi FOLLOW_HOLD_HYSTERESIS: dừng đúng FOLLOW_DISTANCE_MM; đã dừng thì chỉ tiến lại khi xa hơn FOLLOW_DISTANCE_MM + FOLLOW_RESUME_MM,
// và chỉ lùi khi gần hơn FOLLOW_DISTANCE_MM - FOLLOW_REVERSE_MM (chống tiến/lùi liên tục do box nhiễu).
constexpr float FOLLOW_RESUME_MM = 30.0f;
constexpr float FOLLOW_REVERSE_MM = 40.0f;
constexpr bool ALLOW_REVERSE_WHEN_CLOSE = true;        // true = lùi khi tiền quá gần; false = chỉ dừng

// Dừng theo kích thước Bounding Box khi giơ tiền cao (VL53L0X không bắn trúng).
// boxRatio = max(w,h) / bề rộng ảnh (tối đa ~1.0). Với 0.85: tiến khi boxRatio < ~0.77 (123 px), dừng trong ~0.77..0.94, lùi khi > ~0.94 (151 px).
// Công thức: dừng/lùi theo khoảng cách ước lượng = FOLLOW_DISTANCE * BOX_RATIO_AT_FOLLOW_DIST / boxRatio; chỉnh lại khi chạy thực tế:
// - Xe dừng khi còn quá xa: TĂNG BOX_RATIO_AT_FOLLOW_DIST.
// - Xe tiến quá sát mới dừng: GIẢM BOX_RATIO_AT_FOLLOW_DIST.
constexpr float BOX_RATIO_AT_FOLLOW_DIST = 0.85f;
constexpr float BOX_RATIO_FILTER_ALPHA   = 0.50f;      // hệ số lọc khoảng cách ước lượng từ box khi có frame mới (0..1)

constexpr float KP_DIST = 1.8f;                        // vận tốc (mm/s) = KP_DIST * (sai lệch khoảng cách mm)
constexpr float KP_DIST_BACK = 6.0f;                   // như KP_DIST nhưng cho chiều LÙI (box không to hơn ảnh nên cần hệ số lớn hơn)
constexpr float V_FWD_MAX_MM_S = 140.0f;               // tốc độ tiến tối đa khi bám
constexpr float V_FWD_MIN_MM_S = 25.0f;                // tốc độ tiến tối thiểu khi còn đang tiến (tránh kẹt ở vùng chết động cơ)
constexpr float V_BACK_MAX_MM_S = 50.0f;               // tốc độ lùi tối đa (chỉ dùng khi ALLOW_REVERSE_WHEN_CLOSE = true)
constexpr float V_APPROACH_MM_S = 55.0f;               // tốc độ tiến thăm dò khi chưa rõ khoảng cách

constexpr float CAMERA_HFOV_DEG = 65.0f;               // góc nhìn ngang của camera OV2640
constexpr int BEARING_SIGN = +1;                       // đổi thành -1 nếu tiền ở bên phải ảnh mà xe lại quay trái
constexpr float YAW_TARGET_FILTER_ALPHA = 0.65f;       // làm mượt góc đích yawT giữa các frame (0..1): thấp = ít lắc khi box nhiễu, nhưng bám tiền đang di chuyển chậm hơn

constexpr float ROTATE_ONLY_DEG = 45.0f;               // lệch hơn mức này: xoay tại chỗ hướng về tiền trước khi tiến
constexpr float TOF_USE_BEARING_DEG = 20.0f;           // lệch hơn mức này: tốc độ tiến giảm tuyến tính về 0 tại ROTATE_ONLY_DEG
constexpr float HEAD_KP = 6.0f;                         // omega (do/s) = HEAD_KP * (|sai_lech_huong| - HEAD_TOL_DEG) - HEAD_KD * (tốc độ đang quay về phía đích)
constexpr float HEAD_KD = 0.15f;                       // giảm chấn theo tốc độ góc gyro (chống lố khi HEAD_KP / MAX_TURN_DPS cao)
constexpr float HEAD_TOL_DEG = 4.0f;                   // lệch dưới mức này coi như đã thẳng hướng
constexpr float MAX_TURN_DPS = 120.0f;                 // tốc độ quay tối đa

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
constexpr float PWM_SLEW_PER_S = 700.0f;               // giới hạn tốc độ đổi PWM đầu ra (PWM/s): 7 đơn vị mỗi chu kỳ 10 ms. Giảm dòng đỉnh/sụt áp; đảo chiều đi qua 0 trong ~0.1 s
constexpr float FF_KNEE_MM_S = 10.0f;                  // dưới tốc độ này feed-forward tăng tuyến tính từ 0 (không nhảy 0 -> PWM_DEADBAND khi đổi chiều)
constexpr float ACCEL_MM_S2 = 900.0f;                // gia tốc tối đa của vận tốc đích bánh (thấp = êm nhưng quay/đảo chiều chậm: 800 mất 0.26 s để lên 209 mm/s)

// ============================== MPU6050 ======================================
constexpr float MPU_YAW_MOUNT_DEG = 45.0f;             // MPU6050 nằm ngang, lệch 45 độ so với phương di chuyển
constexpr int GYRO_CAL_SAMPLES = 150;                  // 150 mẫu x 10ms = 1.5 s đứng yên lúc khởi động
constexpr float GYRO_CAL_MAX_RANGE_DPS = 3.0f;
constexpr uint32_t GYRO_RETRY_MS = 1000;

// Chống trôi góc Yaw khi xe đứng yên (Zero-Velocity Update - ZUPT):
constexpr int ZUPT_HOLD_TICKS = 8;                     // 8 chu kỳ x 10ms = 80ms cả 2 bánh không quay -> kích hoạt ZUPT
constexpr float ZUPT_MAX_RATE_DPS = 2.5f;              // ngưỡng tốc độ góc tối đa coi là nhiễu trôi khi đứng yên
constexpr float GYRO_BIAS_ADAPT_ALPHA = 0.005f;        // tốc độ tự thích nghi bias nhiệt khi xe đứng yên


// ============================== CHỐNG TREO (sau sụt áp / nhiễu I2C) ==========
constexpr int IMU_FROZEN_TICKS = 30;                   // 30 chu kỳ x 10 ms = 0.3 s MPU6050 trả đúng y hệt một mẫu -> coi là đóng băng (ngủ/mất I2C)
constexpr uint32_t IMU_REINIT_MS = 300;                // IMU xấu liên tục chừng này ms thì khởi tạo lại I2C + MPU6050
constexpr uint32_t IMU_REINIT_RETRY_MS = 1500;         // tối thiểu giữa 2 lần khởi tạo lại IMU
constexpr uint32_t TOF_REINIT_MS = 1500;               // VL53L0X không có mẫu mới chừng này ms thì khởi tạo lại (chỉ khi xe đang đứng yên, vì mất ~100 ms)
constexpr uint32_t TOF_REINIT_RETRY_MS = 3000;         // tối thiểu giữa 2 lần khởi tạo lại ToF

}  // namespace Cfg