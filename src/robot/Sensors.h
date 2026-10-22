#pragma once
#include <stdint.h>

namespace Sensors {

struct Reading {
    float yawDeg = 0;
    float rateDps = 0;
    bool imuGood = false;
    uint16_t tofMm = 9999;
    uint32_t tofAgeMs = 0xFFFFFFFFu;
    int32_t encL = 0, encR = 0;
};

void begin();
bool imuPresent();
bool tofPresent();
bool calibrateGyro();
void update(float dt, Reading& out);

}  // namespace Sensors