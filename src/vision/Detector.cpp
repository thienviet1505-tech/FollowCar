#include "Detector.h"
#include <Arduino.h>
#include <algorithm>
#include <cmath>
#include "RobotConfig.h"

namespace {
// Số class KHÔNG đặt cứng ở đây nữa: lấy từ output tensor [1, 4 + số_class, số_box].
constexpr float CONF_THRESHOLD = Cfg::DETECT_CONF_THRESHOLD;   // chỉnh trong RobotConfig.h
constexpr float NMS_IOU_THRESHOLD = 0.45f;

float iou(const Detection& a, const Detection& b) {
    int ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
    int ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
    int bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
    int bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

    int ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    int ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);

    int iw = std::max(0, ix2 - ix1), ih = std::max(0, iy2 - iy1);
    int inter = iw * ih;
    int uni = a.w * a.h + b.w * b.h - inter;

    return uni > 0 ? (float)inter / (float)uni : 0.0f;
}

std::vector<Detection> nonMaxSuppression(std::vector<Detection>& candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;

        kept.push_back(candidates[i]);

        for (size_t j = i + 1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;

            if (candidates[i].class_id == candidates[j].class_id &&
                iou(candidates[i], candidates[j]) > NMS_IOU_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

std::vector<float> g_classMax;  // điểm cao nhất của từng class ở frame gần nhất
}  // namespace

void Detector::decode(const ModelInference& model, int origWidth, int origHeight,
                      std::vector<Detection>& outDetections) {
    outDetections.clear();
    g_classMax.clear();

    int numDims = model.outputNumDims();

    if (numDims < 3) {
        Serial.println("[Detector] Output tensor khong dung shape mong doi, kiem tra lai model.");
        return;
    }

    // Output YOLOv8: [1, 4 + số_class, số_box] -> số class = số hàng - 4.
    // (Trước đây đặt cứng 6 trong khi model chỉ có 4 class nên đọc lố ra ngoài tensor.)
    int numClasses = model.outputDim(numDims - 2) - 4;
    if (numClasses < 1) {
        Serial.printf("[Detector] Output chi co %d hang, can it nhat 5 (4 toa do + 1 class)\n",
                      model.outputDim(numDims - 2));
        return;
    }
    g_classMax.assign(numClasses, 0.0f);

    const float* out = model.outputData();

    int numBoxes = model.outputDim(numDims - 1);
    int rowStride = numBoxes;

    std::vector<Detection> candidates;
    candidates.reserve(64);

    bool hasNan = false;

    for (int j = 0; j < numBoxes; j++) {
        float cx = out[0 * rowStride + j];
        float cy = out[1 * rowStride + j];
        float w  = out[2 * rowStride + j];
        float h  = out[3 * rowStride + j];

        int bestClass = -1;
        float bestScore = 0.0f;

        for (int c = 0; c < numClasses; c++) {
            float s = out[(4 + c) * rowStride + j];

            if (std::isnan(s)) {
                hasNan = true;
                continue;
            }

            if (s > g_classMax[c]) g_classMax[c] = s;

            if (s > bestScore) {
                bestScore = s;
                bestClass = c;
            }
        }

        if (bestClass >= 0 && bestScore >= CONF_THRESHOLD) {
            const int modelW = model.inputWidth();
            const int modelH = model.inputHeight();

            // Tính lại đúng letterbox parameters giống preprocessFrame().
            float scaleW = (float)modelW / (float)origWidth;
            float scaleH = (float)modelH / (float)origHeight;
            float scale = std::min(scaleW, scaleH);

            int resizeW = (int)std::round(origWidth * scale);
            int resizeH = (int)std::round(origHeight * scale);

            float padXf = (modelW - resizeW) / 2.0f;
            float padYf = (modelH - resizeH) / 2.0f;

            int padX = (int)std::round(padXf - 0.1f);
            int padY = (int)std::round(padYf - 0.1f);

            // Output model đã normalize theo kích thước input: x, w chia cho modelW; y, h chia cho modelH.
            float cxModel = cx * modelW;
            float cyModel = cy * modelH;
            float wModel = w * modelW;
            float hModel = h * modelH;

            // xyxy trên ảnh letterbox.
            float x1 = cxModel - wModel * 0.5f;
            float y1 = cyModel - hModel * 0.5f;
            float x2 = cxModel + wModel * 0.5f;
            float y2 = cyModel + hModel * 0.5f;

            // Bỏ phần padding và đưa về ảnh camera.
            x1 = (x1 - padX) / scale;
            y1 = (y1 - padY) / scale;
            x2 = (x2 - padX) / scale;
            y2 = (y2 - padY) / scale;

            // Giới hạn box trong ảnh gốc.
            x1 = std::max(0.0f, std::min(x1, (float)origWidth));
            y1 = std::max(0.0f, std::min(y1, (float)origHeight));
            x2 = std::max(0.0f, std::min(x2, (float)origWidth));
            y2 = std::max(0.0f, std::min(y2, (float)origHeight));

            float boxW = x2 - x1;
            float boxH = y2 - y1;

            if (boxW <= 0.0f || boxH <= 0.0f) continue;

            Detection d;
            d.class_id = bestClass;
            d.score = bestScore;
            d.cx = (int)std::round((x1 + x2) * 0.5f);
            d.cy = (int)std::round((y1 + y2) * 0.5f);
            d.w = (int)std::round(boxW);
            d.h = (int)std::round(boxH);

            candidates.push_back(d);
        }
    }
    if (hasNan) {
        Serial.println("[Detector] CANH BAO: Output tensor chua gia tri NaN!");
    }

    outDetections = nonMaxSuppression(candidates);
}

const std::vector<float>& Detector::classMaxScores() {
    return g_classMax;
}