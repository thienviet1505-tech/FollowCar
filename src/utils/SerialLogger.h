#pragma once
#include <vector>
#include "Detector.h"

class SerialLogger {
public:
    static void begin(unsigned long baud = 115200);
    static void logFrameResult(const std::vector<Detection>& detections, int imgWidth, int imgHeight);
};
