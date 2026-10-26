#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "Detector.h"
#include "RobotConfig.h"
#include "RobotMath.h"

// Chọn tờ tiền mục tiêu (TARGET_CLASS) từ danh sách box của MỘT frame, giữ trạng thái giữa các frame.
//
// - Chưa bám: chỉ nhận box có điểm >= DETECT_ACQUIRE_THRESH; đang bám: box ngoài cổng cần >= DETECT_HOLD_THRESH
//   (box to nhất thắng, như bản gốc).
// - DETECT_GATE: box trong "cổng" quanh vị trí dự kiến (hướng đã bù góc xe quay, cạnh lớn gần lần trước) được nhận ở ngưỡng
//   thấp hơn và thắng box ngoài cổng; trong cổng chọn box ĐIỂM CAO NHẤT. Hụt tối đa GATE_MAX_MISS_FRAMES frame vẫn giữ cổng.
// - DETECT_SCORE_SMOOTH (3a): giữ điểm EMA của mục tiêu đang bám. Điểm EMA cao -> ngưỡng trong cổng hạ xuống
//   DETECT_GATE_MIN_THRESH (frame tụt conf không làm mất mục tiêu); điểm EMA thấp -> ngưỡng về DETECT_HOLD_THRESH.
//   Nhãn lớp vẫn quyết định: chỉ box thuộc TARGET_CLASS được xét.
class TargetSelector {
public:
    struct Pick {
        bool found = false;
        float bearingDeg = 0.0f;   // + = tiền lệch về bên trái
        float boxRatio = 0.0f;     // cạnh lớn / bề rộng ảnh
    };

    // yawNow: yaw hiện tại của xe (độ), dùng để bù góc xe quay giữa hai frame.
    Pick select(const std::vector<Detection>& dets, int origW, float yawNow) {
        const float expBearing = bearing_ - (yawNow - yaw_);   // xe quay trái +D thì tiền (trái = +) lệch vào giữa -D
        // Vừa buông mục tiêu (hụt quá GATE_MAX_MISS_FRAMES): trong REACQUIRE_GRACE_FRAMES frame kế bắt lại bằng ngưỡng giữ, không đòi ngưỡng bắt mới.
        const bool grace = !holding_ && graceFrames_ > 0;
        const float outsideThresh = (holding_ || grace) ? Cfg::DETECT_HOLD_THRESH : Cfg::DETECT_ACQUIRE_THRESH;
        const float gateThresh = gateThreshold();

        const Detection* bestGate = nullptr;
        const Detection* bestOther = nullptr;
        float scoreGate = -1.0f, areaOther = -1.0f, brgGate = 0.0f, brgOther = 0.0f;

        for (const auto& d : dets) {
            if (d.class_id != Cfg::TARGET_CLASS) continue;
            const float brg = RM::bearingDeg((float)d.cx, (float)origW);
            const float side = (float)std::max(d.w, d.h);
            const bool inGate = Cfg::DETECT_GATE && holding_ &&
                                std::fabs(brg - expBearing) <= Cfg::GATE_BEARING_DEG &&
                                side >= side_ * Cfg::GATE_SIZE_MIN && side <= side_ * Cfg::GATE_SIZE_MAX;
            if (d.score < (inGate ? gateThresh : outsideThresh)) continue;
            if (inGate) {
                // Trong cổng: box điểm cao nhất (box nhiễu conf thấp nhưng to hơn không được thắng)
                if (d.score > scoreGate) { scoreGate = d.score; bestGate = &d; brgGate = brg; }
            } else {
                const float area = (float)(d.w * d.h);
                if (area > areaOther) { areaOther = area; bestOther = &d; brgOther = brg; }
            }
        }

        Pick p;
        const Detection* best = bestGate ? bestGate : bestOther;
        if (best) {
            if (bestGate) {
                fused_ = Cfg::SCORE_EMA_ALPHA * best->score + (1.0f - Cfg::SCORE_EMA_ALPHA) * fused_;
            } else {
                fused_ = best->score;   // bắt mới, hoặc box ngoài cổng: coi như bắt lại từ đầu
            }
            holding_ = true;
            miss_ = 0;
            graceFrames_ = 0;
            bearing_ = bestGate ? brgGate : brgOther;
            side_ = (float)std::max(best->w, best->h);
            yaw_ = yawNow;
            p.found = true;
            p.bearingDeg = bearing_;
            p.boxRatio = (origW > 0) ? (side_ / (float)origW) : 0.0f;
        } else {
            fused_ *= (1.0f - Cfg::SCORE_EMA_ALPHA);   // hụt = điểm 0 trong EMA
            const int maxMiss = Cfg::DETECT_GATE ? Cfg::GATE_MAX_MISS_FRAMES : 0;
            if (holding_) {
                if (++miss_ > maxMiss) {
                    reset();
                    graceFrames_ = Cfg::REACQUIRE_GRACE ? Cfg::REACQUIRE_GRACE_FRAMES : 0;
                }
            } else if (graceFrames_ > 0) {
                graceFrames_--;
            }
        }
        return p;
    }

    void reset() { holding_ = false; fused_ = 0.0f; miss_ = 0; graceFrames_ = 0; }
    bool holding() const { return holding_; }
    float fusedScore() const { return fused_; }

private:
    // Ngưỡng cho box trong cổng: cố định DETECT_GATE_THRESH, hoặc (3a) trượt từ DETECT_HOLD_THRESH (điểm EMA <= SCORE_STRONG_LOW)
    // xuống DETECT_GATE_MIN_THRESH (điểm EMA >= SCORE_STRONG_HIGH).
    float gateThreshold() const {
        if (!Cfg::DETECT_SCORE_SMOOTH) return Cfg::DETECT_GATE_THRESH;
        const float t = RM::clampf((fused_ - Cfg::SCORE_STRONG_LOW) / (Cfg::SCORE_STRONG_HIGH - Cfg::SCORE_STRONG_LOW),
                                   0.0f, 1.0f);
        return Cfg::DETECT_HOLD_THRESH - (Cfg::DETECT_HOLD_THRESH - Cfg::DETECT_GATE_MIN_THRESH) * t;
    }

    bool holding_ = false;
    float bearing_ = 0.0f, side_ = 0.0f, yaw_ = 0.0f, fused_ = 0.0f;
    int miss_ = 0;
    int graceFrames_ = 0;   // > 0: vừa buông mục tiêu, còn được bắt lại bằng ngưỡng giữ
};
