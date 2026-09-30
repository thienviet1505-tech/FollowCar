#include "Detector.h"
#include <Arduino.h>
#include <algorithm>

namespace {
constexpr int NUM_CLASSES = 6;
constexpr float CONF_THRESHOLD = 0.5f;
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
}  // namespace

void Detector::decode(const ModelInference& model, int origWidth, int origHeight,
                      std::vector<Detection>& outDetections) {
    outDetections.clear();

    int numDims = model.outputNumDims();

    if (numDims < 3) {
        Serial.println("[Detector] Output tensor khong dung shape mong doi, kiem tra lai model.");
        return;
    }

    const float* out = model.outputData();

    int numBoxes = model.outputDim(numDims - 1);
    int rowStride = numBoxes;

    std::vector<Detection> candidates;
    candidates.reserve(64);

    for (int j = 0; j < numBoxes; j++) {
        float cx = out[0 * rowStride + j];
        float cy = out[1 * rowStride + j];
        float w  = out[2 * rowStride + j];
        float h  = out[3 * rowStride + j];

        int bestClass = -1;
        float bestScore = 0.0f;

        for (int c = 0; c < NUM_CLASSES; c++) {
            float s = out[(4 + c) * rowStride + j];

            if (s > bestScore) {
                bestScore = s;
                bestClass = c;
            }
        }

        if (bestClass >= 0 && bestScore >= CONF_THRESHOLD) {
            Detection d;
            d.class_id = bestClass;
            d.score = bestScore;
            d.cx = (int)(cx * origWidth);
            d.cy = (int)(cy * origHeight);
            d.w  = (int)(w * origWidth);
            d.h  = (int)(h * origHeight);
            candidates.push_back(d);
        }
    }

    outDetections = nonMaxSuppression(candidates);
}