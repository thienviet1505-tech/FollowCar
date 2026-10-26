#include "BleLog.h"

#if ROBOT_BLE_LOG

#include <Arduino.h>
#include <NimBLEDevice.h>
#include <esp_heap_caps.h>
#include <stdarg.h>
#include <stdio.h>
#include "LogRing.h"

namespace {

constexpr const char* kDeviceName = "FollowRobot";
// Nordic UART Service: các app BLE UART (Serial Bluetooth Terminal, nRF Toolbox...) nhận ra ngay.
constexpr const char* kUuidService = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr const char* kUuidRx = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";   // điện thoại -> xe (app cần có, xe bỏ qua dữ liệu)
constexpr const char* kUuidTx = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";   // xe -> điện thoại (notify)

constexpr size_t kRingBytes = 4096;        // bộ đệm log (ưu tiên đặt trong PSRAM)
constexpr size_t kLineMax = 160;           // dài tối đa 1 dòng (kể cả tiền tố thời gian)
constexpr uint32_t kSendPeriodMs = 15;     // nhịp gửi gói notify (MTU lớn)
constexpr uint32_t kSendPeriodSmallMs = 10; // nhịp gửi khi MTU mặc định (gói chỉ 20 B)
constexpr uint32_t kSettleMs = 2500;       // sau khi kết nối đợi chừng này để app kịp bật Notify rồi mới xả bộ đệm
constexpr size_t kDefaultPayload = 20;     // MTU mặc định 23 - 3
constexpr size_t kPendMax = 244;           // trần payload mỗi gói
constexpr int kTaskStack = 4096;

enum class St : uint8_t { OFF, RUNNING };

volatile St s_state = St::OFF;
volatile bool s_reqStop = false;           // yêu cầu task gửi tự dọn dẹp rồi thoát (chỉ dùng khi pause())
volatile bool s_connected = false;
volatile uint16_t s_mtu = 23;
bool s_started = false;                    // begin() đã được gọi -> resume() mới bật lại

portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
LogRing s_ring;
char* s_ringMem = nullptr;
TaskHandle_t s_task = nullptr;
NimBLEServer* s_server = nullptr;
NimBLECharacteristic* s_tx = nullptr;

size_t freeInternal() { return heap_caps_get_free_size(MALLOC_CAP_INTERNAL); }

// Cố ý KHÔNG dùng 'override': nếu bản thư viện có chữ ký khác thì chỉ mất phần tối ưu (MTU nhỏ), không lỗi biên dịch.
// platformio.ini ghim NimBLE-Arduino 1.4.x, chữ ký dưới đây đúng với bản đó.
class ServerCb : public NimBLEServerCallbacks {
public:
    void onMTUChange(uint16_t mtu, ble_gap_conn_desc*) { s_mtu = mtu; }
};
ServerCb s_serverCb;

// Dọn mọi thứ (chạy trong task gửi, hoặc khi khởi động thất bại). Sau hàm này không còn gì chiếm SRAM.
void teardown() {
    s_state = St::OFF;   // log() ngừng nhận ngay
    portENTER_CRITICAL(&s_mux);
    char* mem = s_ringMem;
    s_ringMem = nullptr;
    s_ring.init(nullptr, 0);   // push() thấy buf == nullptr thì bỏ qua -> an toàn dù có luồng khác đang gọi log()
    portEXIT_CRITICAL(&s_mux);
    NimBLEDevice::deinit(true);
    s_server = nullptr;
    s_tx = nullptr;
    if (mem) heap_caps_free(mem);
    s_connected = false;
    s_mtu = 23;
}

void senderTask(void*) {
    uint8_t pend[kPendMax];
    bool wasConn = false;
    uint32_t connectedSince = 0;

    while (!s_reqStop) {
        const int nconn = s_server ? (int)s_server->getConnectedCount() : 0;
        if (nconn == 0) {
            if (wasConn) { wasConn = false; s_connected = false; s_mtu = 23; }
            NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
            if (adv && !adv->isAdvertising()) adv->start();   // chưa có ai kết nối / vừa bị ngắt: phải quảng bá (lại)
            vTaskDelay(pdMS_TO_TICKS(250));
            continue;
        }
        if (!wasConn) {
            wasConn = true;
            s_connected = true;
            connectedSince = millis();
            portENTER_CRITICAL(&s_mux);
            s_ring.droppedLines = 0;   // chỉ báo số dòng bị bỏ kể từ lúc kết nối
            portEXIT_CRITICAL(&s_mux);
        }
        size_t cap = (s_mtu > 23) ? (size_t)(s_mtu - 3) : kDefaultPayload;
        if (cap > kPendMax) cap = kPendMax;
        if (millis() - connectedSince >= kSettleMs) {
            size_t n;
            uint32_t dropped;
            portENTER_CRITICAL(&s_mux);
            n = s_ring.pop(pend, cap);
            dropped = s_ring.droppedLines;
            s_ring.droppedLines = 0;
            portEXIT_CRITICAL(&s_mux);
            if (n > 0) {
                s_tx->setValue(pend, n);
                s_tx->notify();
            }
            if (dropped > 0) BleLog::log("[BLE] bo %u dong cu (bo dem day, ket noi cham)", (unsigned)dropped);
        }
        // MTU mặc định (gói 20 B) cần gửi dày hơn mới đủ băng thông; app nên xin MTU lớn (vd. nRF Connect: Request MTU 247).
        vTaskDelay(pdMS_TO_TICKS(cap <= kDefaultPayload ? kSendPeriodSmallMs : kSendPeriodMs));
    }

    const size_t before = freeInternal();
    teardown();
    Serial.printf("[BLE] TAT | SRAM noi con %u B (giai phong ~%d B)\n", (unsigned)freeInternal(),
                  (int)(freeInternal() - before));
    s_task = nullptr;
    vTaskDelete(nullptr);
}

bool startRunning() {
    if (s_state == St::RUNNING) return true;
    if (s_task != nullptr) {
        Serial.println("[BLE] Dang don dep lan tat truoc, thu lai sau");
        return false;
    }
    const size_t before = freeInternal();

    s_ringMem = (char*)heap_caps_malloc(kRingBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_ringMem) s_ringMem = (char*)heap_caps_malloc(kRingBytes, MALLOC_CAP_8BIT);
    if (!s_ringMem) {
        Serial.println("[BLE] Khong du RAM cho bo dem log");
        return false;
    }
    portENTER_CRITICAL(&s_mux);
    s_ring.init(s_ringMem, kRingBytes);
    portEXIT_CRITICAL(&s_mux);

    NimBLEDevice::init(kDeviceName);
    NimBLEDevice::setPower(ESP_PWR_LVL_N0);   // 0 dBm đủ cho vài mét, giảm xung dòng khi phát
    NimBLEDevice::setMTU(247);

    s_server = NimBLEDevice::createServer();
    s_server->setCallbacks(&s_serverCb);
    NimBLEService* svc = s_server->createService(kUuidService);
    s_tx = svc->createCharacteristic(kUuidTx, NIMBLE_PROPERTY::NOTIFY);
    svc->createCharacteristic(kUuidRx, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
    svc->start();

    NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(kUuidService);
    adv->start();   // phát ngay: điện thoại quét thấy "FollowRobot" là kết nối được

    s_reqStop = false;
    s_connected = false;
    s_mtu = 23;
    s_state = St::RUNNING;   // phải đặt trước khi tạo task để log() nhận dòng

    // Lõi 0 + ưu tiên 1: chỉ chạy bằng thời gian rảnh, không bao giờ làm trễ vòng điều khiển (ưu tiên 20) hay suy luận (lõi 1).
    if (xTaskCreatePinnedToCore(senderTask, "blelog", kTaskStack, nullptr, 1, &s_task, 0) != pdPASS) {
        Serial.println("[BLE] Khong tao duoc task gui");
        s_task = nullptr;
        teardown();
        return false;
    }

    BleLog::log("[BLE] san sang. SRAM noi con %u", (unsigned)freeInternal());
    Serial.printf("[BLE] BAT | dang phat '%s' | SRAM noi con %u B (BLE chiem ~%d B, chua tinh luc dang ket noi)\n",
                  kDeviceName, (unsigned)freeInternal(), (int)(before - freeInternal()));
    return true;
}

void stopRunning() {
    if (s_state != St::RUNNING) return;
    s_reqStop = true;
    const uint32_t t0 = millis();
    while ((s_state != St::OFF || s_task != nullptr) && millis() - t0 < 3000) delay(10);
}

}  // namespace

namespace BleLog {

void begin() {
    s_started = true;
    startRunning();
}

bool enabled() { return s_state == St::RUNNING; }
bool connected() { return s_connected; }

void pause() { stopRunning(); }   // WiFi (VIEW) cần SRAM nội

void resume() { if (s_started) startRunning(); }

void log(const char* fmt, ...) {
    if (s_state != St::RUNNING) return;   // tắt thì không format, không khóa: chi phí bằng 0
    char buf[kLineMax];
    const uint32_t ms = millis();
    const int p = snprintf(buf, sizeof buf, "%lu.%02lu ", (unsigned long)(ms / 1000), (unsigned long)((ms % 1000) / 10));
    if (p < 0) return;
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf + p, sizeof buf - p - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    size_t len = (size_t)p + (size_t)n;
    if (len > sizeof buf - 2) len = sizeof buf - 2;
    buf[len++] = '\n';
    portENTER_CRITICAL(&s_mux);
    s_ring.push(buf, len);
    portEXIT_CRITICAL(&s_mux);
}

}  // namespace BleLog

#endif  // ROBOT_BLE_LOG
