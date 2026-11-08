#include "ViewMode.h"
#include <Arduino.h>
#if ROBOT_VIEW_MODE
#include <WiFi.h>
#endif
#include <esp_heap_caps.h>
#include "Robot.h"
#include "BleLog.h"

namespace {

bool s_active = false;

void printMem(const char* tag) {
    Serial.printf("[Mem] %s | SRAM trong: free=%u, khoi lon nhat=%u | PSRAM free=%u\n", tag,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

void printHelp() {
#if ROBOT_VIEW_MODE
    Serial.println("[Lenh] v = xem anh camera tren trinh duyet (bat WiFi, tam dung nhan dien)");
    Serial.println("[Lenh] d = ve che do nhan dien (tat WiFi)");
#else
    Serial.println("[Lenh] (che do VIEW da tat luc build: ROBOT_VIEW_MODE = 0 trong RobotConfig.h)");
#endif
    Serial.println("[Lenh] t = chay lai self-test voi anh mau to 100k");
    Serial.println("[Lenh] p = in thong so camera hien tai de dan vao CameraModule.cpp");
    Serial.println("[Lenh] m = in RAM con trong    ? = tro giup");
#if ROBOT_VIEW_MODE
    if (Cfg::BUTTON_ENABLE) Serial.println("[Nut ] G0 (BOOT) = doi qua lai giua v va d (xe tu dung khi vao che do xem anh)");
#endif
}

#if ROBOT_VIEW_MODE

void startView() {
    IPAddress ip;
    bool haveIp = false;

    if (strlen(VIEW_WIFI_SSID) > 0) {
        Serial.printf("[View] Noi WiFi '%s'", VIEW_WIFI_SSID);
        WiFi.mode(WIFI_STA);
        WiFi.begin(VIEW_WIFI_SSID, VIEW_WIFI_PASSWORD);
        WiFi.setSleep(false);
        uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - t0 < VIEW_STA_TIMEOUT_MS) {
            delay(250);
            Serial.print('.');
        }
        Serial.println();
        if (WiFi.status() == WL_CONNECTED) {
            ip = WiFi.localIP();
            haveIp = true;
        } else {
            Serial.println("[View] Khong noi duoc WiFi, chuyen sang phat WiFi rieng");
            WiFi.disconnect(true, false);
        }
    }

    if (!haveIp) {
        WiFi.mode(WIFI_AP);
        if (!WiFi.softAP(VIEW_AP_SSID, VIEW_AP_PASSWORD)) {
            Serial.println("[View] Phat WiFi rieng that bai");
            WiFi.mode(WIFI_OFF);
            return;
        }
        ip = WiFi.softAPIP();
        Serial.printf("[View] AP '%s' da san sang\n", VIEW_AP_SSID);
    }

    startCameraServer();
    s_active = true;

    Serial.printf("[View] Mo trinh duyet: http://%s   (luong anh o cong 81)\n", ip.toString().c_str());
    Serial.println("[View] Chinh thanh truot tren web, xong go 'p' de lay thong so, go 'd' de ve nhan dien");
    Serial.println("[View] Chup dataset: bam Start Stream, roi nut 'Chup anh' (hoac Space), xem lai, bam 'Tai ZIP'");
}

void stopView(CameraModule& camera) {
    stopCameraServer();
    WiFi.softAPdisconnect(true);
    WiFi.disconnect(true, false);
    WiFi.mode(WIFI_OFF);
    s_active = false;

    // frame trong bo dem camera la frame cu tu luc xem anh -> bo di
    camera.discardFrames(3);
    Serial.println("[Mode] DETECT");
}

// Vào/ra chế độ xem ảnh: xe phải dừng trước khi bật WiFi, và chỉ được chạy lại sau khi về DETECT.
void enterView() {
    Robot::setEnabled(false);
    BleLog::pause();                           // nhường SRAM cho WiFi
    startView();
    if (!s_active) {                           // bật WiFi thất bại -> ở lại DETECT
        Robot::setEnabled(true);
        BleLog::resume();
    }
}

void leaveView(CameraModule& camera) {
    stopView(camera);
    BleLog::resume();
    Robot::setEnabled(true);
}
#else
void enterView() { Serial.println("[View] Da tat luc build (ROBOT_VIEW_MODE = 0 trong RobotConfig.h)"); }
void leaveView(CameraModule&) {}
#endif

}  // namespace

namespace ViewMode {

namespace {
bool s_selfTestRequested = false;
}

void requestSelfTest() { s_selfTestRequested = true; }

bool takeSelfTestRequest() {
    const bool r = s_selfTestRequested;
    s_selfTestRequested = false;
    return r;
}

bool active() { return s_active; }

void handleSerial(CameraModule& camera) {
    // Nut G0: doi qua lai v <-> d (task dieu khien da tu dung xe ngay luc bam)
    if (Robot::takeButtonPress()) {
        if (s_active) leaveView(camera);
        else enterView();
    }
    while (Serial.available()) {
        char c = (char)Serial.read();
        switch (c) {
            case 'v': case 'V':
                if (s_active) Serial.println("[Mode] Dang o che do VIEW roi");
                else enterView();
                break;
            case 'd': case 'D':
                if (s_active) leaveView(camera);
                else Serial.println("[Mode] Dang o che do DETECT roi");
                break;
            case 't': case 'T':
                requestSelfTest();
                break;
            case 'p': case 'P':
                camera.printSettings();
                break;
            case 'm': case 'M':
                printMem("hien tai");
                break;
            case '?': case 'h': case 'H':
                printHelp();
                break;
            default:
                break;  // bo qua \r \n va ky tu khac
        }
    }
}

}  // namespace ViewMode
