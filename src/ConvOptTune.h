#pragma once

// Tự đo tốc độ các conv 3x3 (ConvOpt) trên chính board: chạy thử từng cách (kernel ESP-NN gốc,
// kernel ConvOpt một lõi, hai lõi chia theo kênh ra, hai lõi chia theo điểm ảnh), so kết quả
// từng bit với kernel gốc, rồi chọn cách nhanh nhất cho từng conv.
// Dùng: ConvOptTuneBegin(); interpreter->Invoke(); ConvOptTuneEnd();  (một lần, sau AllocateTensors
// và esp_nn_dual_core_enable). Không gọi thì mọi conv chạy bằng kernel gốc.
void ConvOptTuneBegin();
void ConvOptTuneEnd();
void ConvOptPrintSummary();
