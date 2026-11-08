#include "SerialLogger.h"
#include <Arduino.h>

void SerialLogger::begin(unsigned long baud) {
    Serial.begin(baud);
    unsigned long t0 = millis();
    while (!Serial && millis() - t0 < 3000) { delay(10); }
}
