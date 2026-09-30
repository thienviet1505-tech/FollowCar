#include "SerialLogger.h"
#include <Arduino.h>
#include <set>

void SerialLogger::begin(unsigned long baud) {
    Serial.begin(baud);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 3000) { delay(10); }
}

void SerialLogger::logFrameResult(const std::vector<Detection>& detections, int imgWidth, int imgHeight) {
    if (detections.empty()) {
        Serial.println("[Detect] classes=0");
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

    Serial.printf("[Detect] classes=%d | largest: class=%d score=%.2f area=%dpx^2 offset=(%d,%d)px\n",
                  (int)uniqueClasses.size(), largest->class_id, largest->score,
                  largest->area(), offsetX, offsetY);
}
