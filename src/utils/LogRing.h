#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Bộ đệm vòng chứa các dòng log (mỗi dòng kết thúc bằng '\n').
// Đầy thì bỏ DÒNG CŨ NHẤT để nhường chỗ (luôn giữ được những dòng mới nhất, không bao giờ chặn người ghi).
// Không tự khóa: người dùng phải bao push()/pop() bằng khóa của mình (xem BleLog.cpp).
struct LogRing {
    char* buf = nullptr;
    size_t cap = 0, head = 0, tail = 0, used = 0;
    uint32_t droppedLines = 0;

    void init(char* b, size_t c) { buf = b; cap = c; head = tail = used = 0; droppedLines = 0; }

    // d: n byte, nên kết thúc bằng '\n'. Trả false nếu chưa init hoặc dòng dài hơn cả bộ đệm.
    bool push(const char* d, size_t n) {
        if (!buf || n == 0 || n > cap) return false;
        while (cap - used < n) {
            // bỏ dòng cũ nhất: đi tới hết '\n' đầu tiên (hoặc hết dữ liệu)
            while (used > 0) {
                const char c = buf[tail];
                tail = (tail + 1) % cap;
                used--;
                if (c == '\n') break;
            }
            droppedLines++;
        }
        const size_t first = (n < cap - head) ? n : cap - head;   // đoạn tới cuối bộ đệm
        memcpy(buf + head, d, first);
        memcpy(buf, d + first, n - first);
        head = (head + n) % cap;
        used += n;
        return true;
    }

    // Lấy tối đa max byte cũ nhất ra out.
    size_t pop(uint8_t* out, size_t max) {
        const size_t n = used < max ? used : max;
        const size_t first = (n < cap - tail) ? n : cap - tail;
        memcpy(out, buf + tail, first);
        memcpy(out + first, buf, n - first);
        tail = (tail + n) % cap;
        used -= n;
        return n;
    }
};
