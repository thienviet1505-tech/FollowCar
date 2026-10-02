#include "CameraModule.h"
#include <Arduino.h>

// =============================================================================
// BẢNG THÔNG SỐ MẶC ĐỊNH THEO TỪNG CẢM BIẾN
// -----------------------------------------------------------------------------
// Cách dùng: chỉnh thanh trượt trên trình duyệt (gõ 'v' qua Serial), ưng ý thì
// gõ 'p' — chương trình in ra các dòng "t.xxx = ...;" — dán đè vào hàm của đúng
// cảm biến bên dưới. Trường nào để KEEP nghĩa là không đụng (giữ mặc định của
// driver). Khoảng giá trị giống thanh trượt trên trang web:
//   brightness/contrast/saturation/sharpness/ae_level: -2..2
//   awb, awb_gain, aec, aec2, agc, bpc, wpc, raw_gma, lenc, hmirror, vflip, dcw: 0/1
//   wb_mode: 0..4     aec_value: 0..1200 (chỉ có tác dụng khi aec = 0)
//   agc_gain: 0..30 (chỉ có tác dụng khi agc = 0)   gainceiling: 0..6 (2x..128x)
// =============================================================================
namespace {
constexpr int KEEP = -1000;

struct SensorTuning {
    int awb = KEEP, awb_gain = KEEP, wb_mode = KEEP;
    int aec = KEEP, aec2 = KEEP, ae_level = KEEP, aec_value = KEEP;
    int agc = KEEP, agc_gain = KEEP, gainceiling = KEEP;
    int brightness = KEEP, contrast = KEEP, saturation = KEEP, sharpness = KEEP;
    int bpc = KEEP, wpc = KEEP, raw_gma = KEEP, lenc = KEEP;
    int hmirror = KEEP, vflip = KEEP, dcw = KEEP;
};

// OV3660: giữ nguyên mặc định của driver (đang chạy tốt).
SensorTuning makeOV3660() {
    SensorTuning t;
    return t;
}

// OV2640: kém nhạy sáng hơn OV3660 nên ảnh tối. Điểm xuất phát: bật AEC nâng
// cao, bù sáng +1, trần gain 8x. Tinh chỉnh thêm bằng chế độ xem ảnh.
SensorTuning makeOV2640() {
    SensorTuning t;
    t.aec2 = 1;
    t.ae_level = 1;
    t.brightness = 1;
    t.gainceiling = GAINCEILING_8X;
    return t;
}

// Thứ tự áp dụng quan trọng: bật/tắt chế độ tự động trước, rồi mới tới giá trị.
void applyTuning(sensor_t* s, const SensorTuning& t) {
#define APPLY(field, setter) \
    if (t.field != KEEP && s->setter) s->setter(s, t.field)
    APPLY(awb, set_whitebal);
    APPLY(awb_gain, set_awb_gain);
    APPLY(wb_mode, set_wb_mode);
    APPLY(aec, set_exposure_ctrl);
    APPLY(aec2, set_aec2);
    APPLY(ae_level, set_ae_level);
    APPLY(aec_value, set_aec_value);
    APPLY(agc, set_gain_ctrl);
    APPLY(agc_gain, set_agc_gain);
    if (t.gainceiling != KEEP && s->set_gainceiling) {
        s->set_gainceiling(s, (gainceiling_t)t.gainceiling);
    }
    APPLY(brightness, set_brightness);
    APPLY(contrast, set_contrast);
    APPLY(saturation, set_saturation);
    APPLY(sharpness, set_sharpness);
    APPLY(bpc, set_bpc);
    APPLY(wpc, set_wpc);
    APPLY(raw_gma, set_raw_gma);
    APPLY(lenc, set_lenc);
    APPLY(hmirror, set_hmirror);
    APPLY(vflip, set_vflip);
    APPLY(dcw, set_dcw);
#undef APPLY
}
}  // namespace

void CameraModule::applySensorDefaults() {
    sensor_t* s = esp_camera_sensor_get();
    if (!s) return;

    if (s->id.PID == OV2640_PID) {
        applyTuning(s, makeOV2640());
        Serial.println("[CameraModule] Sensor OV2640 - da ap bang thong so OV2640");
    } else if (s->id.PID == OV3660_PID) {
        applyTuning(s, makeOV3660());
        Serial.println("[CameraModule] Sensor OV3660 - da ap bang thong so OV3660");
    } else {
        Serial.printf("[CameraModule] Sensor PID=0x%x - khong co bang thong so rieng\n", s->id.PID);
    }
}

void CameraModule::discardFrames(int n) {
    for (int i = 0; i < n; i++) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) esp_camera_fb_return(fb);
    }
}

void CameraModule::printSettings() {
    sensor_t* s = esp_camera_sensor_get();
    if (!s) {
        Serial.println("[CameraModule] Khong lay duoc sensor");
        return;
    }
    const camera_status_t& st = s->status;
    const char* fn = (s->id.PID == OV2640_PID) ? "makeOV2640()"
                   : (s->id.PID == OV3660_PID) ? "makeOV3660()" : "(sensor khac)";
    Serial.printf("// ---- dan de len bang thong so trong %s (CameraModule.cpp) ----\n", fn);
    Serial.printf("t.awb = %d;\nt.awb_gain = %d;\nt.wb_mode = %d;\n", st.awb, st.awb_gain, st.wb_mode);
    Serial.printf("t.aec = %d;\nt.aec2 = %d;\nt.ae_level = %d;\nt.aec_value = %d;\n",
                  st.aec, st.aec2, st.ae_level, st.aec_value);
    Serial.printf("t.agc = %d;\nt.agc_gain = %d;\nt.gainceiling = %d;\n", st.agc, st.agc_gain, st.gainceiling);
    Serial.printf("t.brightness = %d;\nt.contrast = %d;\nt.saturation = %d;\nt.sharpness = %d;\n",
                  st.brightness, st.contrast, st.saturation, st.sharpness);
    Serial.printf("t.bpc = %d;\nt.wpc = %d;\nt.raw_gma = %d;\nt.lenc = %d;\n", st.bpc, st.wpc, st.raw_gma, st.lenc);
    Serial.printf("t.hmirror = %d;\nt.vflip = %d;\nt.dcw = %d;\n", st.hmirror, st.vflip, st.dcw);
    Serial.println("// ---------------------------------------------------------------");
}

bool CameraModule::begin() {
    config_.ledc_channel = LEDC_CHANNEL_0;
    config_.ledc_timer   = LEDC_TIMER_0;
    config_.pin_d0       = Y2_GPIO_NUM;
    config_.pin_d1       = Y3_GPIO_NUM;
    config_.pin_d2       = Y4_GPIO_NUM;
    config_.pin_d3       = Y5_GPIO_NUM;
    config_.pin_d4       = Y6_GPIO_NUM;
    config_.pin_d5       = Y7_GPIO_NUM;
    config_.pin_d6       = Y8_GPIO_NUM;
    config_.pin_d7       = Y9_GPIO_NUM;
    config_.pin_xclk     = XCLK_GPIO_NUM;
    config_.pin_pclk     = PCLK_GPIO_NUM;
    config_.pin_vsync    = VSYNC_GPIO_NUM;
    config_.pin_href     = HREF_GPIO_NUM;
    config_.pin_sccb_sda = SIOD_GPIO_NUM;
    config_.pin_sccb_scl = SIOC_GPIO_NUM;
    config_.pin_pwdn     = PWDN_GPIO_NUM;
    config_.pin_reset    = RESET_GPIO_NUM;
    config_.xclk_freq_hz = 20000000;
    config_.pixel_format = PIXFORMAT_RGB565;  // main.cpp đọc fb->buf theo định dạng này
    config_.frame_size   = FRAMESIZE_QVGA;    // 320x240, đổi nếu cần ảnh to/nhỏ hơn
    config_.fb_count     = 2;
    config_.fb_location  = CAMERA_FB_IN_PSRAM;
    config_.grab_mode    = CAMERA_GRAB_LATEST;  // luôn lấy frame mới nhất (WHEN_EMPTY có thể trả frame cũ khi Invoke chậm)

    esp_err_t err = esp_camera_init(&config_);
    if (err != ESP_OK) {
        Serial.printf("[CameraModule] esp_camera_init that bai, ma loi: 0x%x\n", err);
        return false;
    }

    applySensorDefaults();
    discardFrames(3);
    return true;
}

camera_fb_t* CameraModule::capture() {
    return esp_camera_fb_get();
}

void CameraModule::release(camera_fb_t* fb) {
    esp_camera_fb_return(fb);
}