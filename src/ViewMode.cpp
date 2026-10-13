#include "ViewMode.h"
#include <Arduino.h>
#include <WiFi.h>
#include <esp_heap_caps.h>

namespace {

bool s_active = false;

void printMem(const char* tag) {
    Serial.printf("[Mem] %s | SRAM trong: free=%u, khoi lon nhat=%u | PSRAM free=%u\n", tag,
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
                  (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

void printHelp() {
    Serial.println("[Lenh] v = xem anh camera tren trinh duyet (bat WiFi, tam dung nhan dien)");
    Serial.println("[Lenh] d = ve che do nhan dien (tat WiFi)");
    Serial.println("[Lenh] t = chay lai self-test voi anh mau to 100k");
    Serial.println("[Lenh] p = in thong so camera hien tai de dan vao CameraModule.cpp");
    Serial.println("[Lenh] m = in RAM con trong    ? = tro giup");
}

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

}  // namespace

extern bool g_triggerSelfTest;

namespace ViewMode {

bool active() { return s_active; }

void handleSerial(CameraModule& camera) {
    while (Serial.available()) {
        char c = (char)Serial.read();
        switch (c) {
            case 'v': case 'V':
                if (s_active) Serial.println("[Mode] Dang o che do VIEW roi");
                else startView();
                break;
            case 'd': case 'D':
                if (s_active) stopView(camera);
                else Serial.println("[Mode] Dang o che do DETECT roi");
                break;
            case 't': case 'T':
                g_triggerSelfTest = true;
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
