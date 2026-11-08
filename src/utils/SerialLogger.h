#pragma once

class SerialLogger {
public:
    // Mở Serial và đợi tối đa 3 s cho cổng USB sẵn sàng.
    static void begin(unsigned long baud = 115200);
};
