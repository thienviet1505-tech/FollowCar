#include "SerialLogger.h"
#include <Arduino.h>
#include <set>

namespace {
// Tên class theo thứ tự id của model (metadata model: 0=100k, 1=10k, 2=50k, 3=5k).
// Sửa lại nếu train lại model với bộ class khác.
const char* const kClassNames[] = {"100k", "10k", "50k", "5k"};
constexpr size_t kNumNames = sizeof(kClassNames) / sizeof(kClassNames[0]);
}  // namespace

void SerialLogger::begin(unsigned long baud) {
    Serial.begin(baud);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 3000) { delay(10); }
}

void SerialLogger::logFrameResult(const std::vector<Detection>& detections, int imgWidth, int imgHeight) {
    // Điểm cao nhất của từng class (kể cả dưới ngưỡng) — xem model đang "nghĩ" gì.
    const std::vector<float>& mx = Detector::classMaxScores();
    if (!mx.empty()) {
        Serial.print("[Score] max:");
        for (size_t c = 0; c < mx.size(); c++) {
            if (c < kNumNames) Serial.printf(" %s=%.3f", kClassNames[c], mx[c]);
            else Serial.printf(" c%u=%.3f", (unsigned)c, mx[c]);
        }
        Serial.println();
    }

    if (detections.empty()) {
        Serial.println("[Detect] none");
        return;
    }

    std::set<int> uniqueClasses;
    const Detection* largest = &detections[0];
    for (const auto& d : detections) {
        uniqueClasses.insert(d.class_id);
        if (d.area() > largest->area()) largest = &d;
    }

    int offsetX = largest->cx - imgWidth / 2;
    int offsetY = largest->cy - imgHeight / 2;

    const char* className = (largest->class_id >= 0 && largest->class_id < (int)kNumNames)
                                ? kClassNames[largest->class_id] : "unknown";

    Serial.printf("[Detect] n=%u | classes=%u | largest=%s score=%.3f box=%dx%d offset=(%+d,%+d)px\n",
                  (unsigned)detections.size(), (unsigned)uniqueClasses.size(),
                  className, largest->score, largest->w, largest->h, offsetX, offsetY);
}