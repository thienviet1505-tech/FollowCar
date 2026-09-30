#!/usr/bin/env python3
"""
Tải và dựng 2 thư viện cho project (chỉ cần chạy 1 lần sau khi tạo project mới):

  lib/TFLiteMicro/   <- espressif/esp-tflite-micro (đã chọn lọc đúng file cần)
  lib/esp-nn/        <- espressif/esp-nn (kernel int8 tối ưu cho ESP32-S3)

Cách chạy (trong thư mục project):
  python tools/setup_libs.py            # bỏ qua thư viện đã có
  python tools/setup_libs.py --force    # xoá và dựng lại từ đầu

Cần internet (tải zip từ github.com). Không cần cài git.
"""
import argparse
import io
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

# Phiên bản đã ghim để build lặp lại được.
TFLM_COMMIT = "dd4085950a119d9e33a6de1d7be1fc7fe29e0cfc"   # esp-tflite-micro 1.4.1
TFLM_URL = "https://github.com/espressif/esp-tflite-micro/archive/%s.zip" % TFLM_COMMIT
NN_TAG = "v1.4.1"
NN_URL = "https://github.com/espressif/esp-nn/archive/refs/tags/%s.zip" % NN_TAG

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB = os.path.join(ROOT, "lib")

# --------------------- danh sách file .cc/.c cần biên dịch ---------------------
# Lấy từ CMakeLists.txt chính thức của esp-tflite-micro, bỏ phần audio/signal.
TFLM_MICRO = ["debug_log", "flatbuffer_utils", "memory_helpers", "micro_allocation_info",
              "micro_allocator", "micro_context", "micro_interpreter_context",
              "micro_interpreter_graph", "micro_interpreter", "micro_log", "micro_op_resolver",
              "micro_profiler", "micro_resource_variable", "micro_utils",
              "recording_micro_allocator", "system_setup"]
# 7 kernel bị thay bởi bản trong kernels/esp_nn/ (đúng như CMakeLists gốc)
ESP_NN_REPLACED = ["add", "conv", "depthwise_conv", "fully_connected", "mul", "pooling", "softmax"]
TFLM_EXPLICIT = [
    "tensorflow/lite/kernels/kernel_util.cc",
    "tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc",
    "tensorflow/lite/micro/memory_planner/linear_memory_planner.cc",
    "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc",
    "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc",
    "tensorflow/lite/core/c/common.cc",
    "tensorflow/lite/core/api/flatbuffer_conversions.cc",
    "tensorflow/lite/core/api/tensor_utils.cc",
    "tensorflow/lite/kernels/internal/common.cc",
    "tensorflow/lite/kernels/internal/quantization_util.cc",
    "tensorflow/lite/kernels/internal/portable_tensor_utils.cc",
    "tensorflow/lite/kernels/internal/tensor_utils.cc",
    "tensorflow/lite/kernels/internal/tensor_ctypes.cc",
    "tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc",
    "tensorflow/compiler/mlir/lite/core/api/error_reporter.cc",
    "tensorflow/compiler/mlir/lite/schema/schema_utils.cc",
]
HEADER_EXT = (".h", ".hpp", ".inc")

# esp-nn: đúng danh sách nguồn của ESP32-S3 trong CMakeLists.txt của esp-nn
ESP_NN_C = [   'src/activation_functions/esp_nn_relu_ansi.c',
    'src/activation_functions/esp_nn_hard_swish_ansi.c',
    'src/common/esp_nn_mean_ansi.c',
    'src/basic_math/esp_nn_add_ansi.c',
    'src/basic_math/esp_nn_mul_ansi.c',
    'src/convolution/esp_nn_conv_ansi.c',
    'src/convolution/esp_nn_conv_opt.c',
    'src/convolution/esp_nn_depthwise_conv_ansi.c',
    'src/convolution/esp_nn_depthwise_conv_opt.c',
    'src/fully_connected/esp_nn_fully_connected_ansi.c',
    'src/softmax/esp_nn_softmax_ansi.c',
    'src/softmax/esp_nn_softmax_opt.c',
    'src/logistic/esp_nn_logistic_ansi.c',
    'src/pooling/esp_nn_avg_pool_ansi.c',
    'src/pooling/esp_nn_max_pool_ansi.c',
    'src/common/esp_nn_multicore.c',
    'src/convolution/esp_nn_conv_mt_split.c',
    'src/basic_math/esp_nn_mul_mt.c']
ESP_NN_S3 = [   'src/common/esp_nn_common_functions_esp32s3.S',
    'src/common/esp_nn_dot_s8_esp32s3.S',
    'src/common/esp_nn_multiply_by_quantized_mult_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s8_kx1_esp32s3.S',
    'src/common/esp_nn_multiply_by_quantized_mult_ver1_esp32s3.S',
    'src/activation_functions/esp_nn_relu_s8_esp32s3.S',
    'src/activation_functions/esp_nn_hard_swish_s8_esp32s3.c',
    'src/common/esp_nn_mean_s8_esp32s3.c',
    'src/basic_math/esp_nn_add_s8_esp32s3.S',
    'src/basic_math/esp_nn_add_mt_esp32s3.c',
    'src/basic_math/esp_nn_mul_s8_esp32s3.S',
    'src/basic_math/esp_nn_mul_broadcast_s8_esp32s3.S',
    'src/convolution/esp_nn_conv_esp32s3.c',
    'src/convolution/esp_nn_conv_s8_1x1_esp32s3.c',
    'src/convolution/esp_nn_conv_s8_3x3_opt_esp32s3.c',
    'src/convolution/esp_nn_depthwise_conv_s8_esp32s3.c',
    'src/convolution/esp_nn_conv_s16_mult8_esp32s3.S',
    'src/convolution/esp_nn_conv_s8_mult8_1x1_esp32s3.S',
    'src/convolution/esp_nn_conv_s16_mult4_1x1_esp32s3.S',
    'src/convolution/esp_nn_conv_s8_filter_aligned_input_padded_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s8_mult1_3x3_padded_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult1_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult1_3x3_no_pad_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult8_3x3_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult4_esp32s3.S',
    'src/convolution/esp_nn_depthwise_conv_s16_mult8_esp32s3.S',
    'src/fully_connected/esp_nn_fully_connected_esp32s3.c',
    'src/fully_connected/esp_nn_fc_s8_mac16_esp32s3.S',
    'src/fully_connected/esp_nn_fully_connected_s8_esp32s3.S',
    'src/fully_connected/esp_nn_fully_connected_per_ch_s8_esp32s3.S',
    'src/pooling/esp_nn_max_pool_s8_esp32s3.S',
    'src/pooling/esp_nn_avg_pool_s8_esp32s3.c',
    'src/pooling/esp_nn_avg_pool_s8_esp32s3.S',
    'src/softmax/esp_nn_softmax_s8_esp32s3.c']

ESP_NN_LIBRARY_JSON = """{
  "name": "esp-nn",
  "version": "1.4.1",
  "description": "Optimized NN kernels for ESP32-S3 (vendored from espressif/esp-nn)",
  "build": {
    "flags": [
      "-Iinclude",
      "-Isrc/common",
      "-O2",
      "-fno-unroll-loops",
      "-Wno-unused-function"
    ],
    "srcDir": "src",
    "includeDir": "include"
  }
}
"""

# Bản gốc Espressif biên dịch riêng file này bằng -O2 vì -O3 từng gây "internal
# compiler error" trong depthwise_conv_float.h. Project dùng -O3 toàn cục nên
# ép -O2 riêng cho file này bằng pragma.
O2_PRAGMA = ("// [setup_libs.py] ep -O2 rieng cho file nay (xem CMakeLists.txt goc cua esp-tflite-micro)\n"
             '#pragma GCC optimize("O2")\n')


def download(url, what):
    print("  tai %s ..." % what, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "setup_libs.py"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    print("  xong (%.1f MB)" % (len(data) / 1e6))
    return zipfile.ZipFile(io.BytesIO(data))


def extract_root(zf, tmp):
    zf.extractall(tmp)
    names = os.listdir(tmp)
    assert len(names) == 1, names
    return os.path.join(tmp, names[0])


def copy_tree_filtered(src_dir, dst_dir, keep):
    """Chép cả cây thư mục, chỉ giữ file mà keep(rel_path) trả True."""
    for dp, _, files in os.walk(src_dir):
        for fn in files:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, src_dir).replace(os.sep, "/")
            if keep(rel):
                out = os.path.join(dst_dir, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(full, out)


def copy_file(src_root, rel, dst_root):
    out = os.path.join(dst_root, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy2(os.path.join(src_root, rel), out)


def setup_tflm(force):
    dst = os.path.join(LIB, "TFLiteMicro")
    if os.path.isdir(dst):
        if not force:
            print("[TFLiteMicro] da co, bo qua (dung --force de dung lai)")
            return
        shutil.rmtree(dst)
    print("[TFLiteMicro] dang dung...")
    with tempfile.TemporaryDirectory() as tmp:
        src = extract_root(download(TFLM_URL, "esp-tflite-micro @ " + TFLM_COMMIT[:8]), tmp)
        d = os.path.join(dst, "src")
        os.makedirs(d)

        # 1) HEADER của tensorflow/lite + các thư mục mlir cần + signal/ (chỉ header)
        for top in ["tensorflow/lite",
                    "tensorflow/compiler/mlir/lite/core/api",
                    "tensorflow/compiler/mlir/lite/core/c",
                    "tensorflow/compiler/mlir/lite/schema",
                    "tensorflow/compiler/mlir/lite/kernels/internal",
                    "signal"]:
            copy_tree_filtered(os.path.join(src, top), os.path.join(d, top),
                               lambda rel: rel.endswith(HEADER_EXT))
        # 2) third_party: chỉ header (gemmlowp, flatbuffers, ruy, kissfft)
        for tp in ["gemmlowp", "flatbuffers", "ruy", "kissfft"]:
            copy_tree_filtered(os.path.join(src, "third_party", tp),
                               os.path.join(d, "third_party", tp),
                               lambda rel: rel.endswith(HEADER_EXT) or
                               os.path.basename(rel) in ("LICENSE", "COPYING"))
        shutil.copy2(os.path.join(src, "LICENSE"), os.path.join(dst, "LICENSE"))

        # 3) file .cc cần biên dịch
        srcs = ["tensorflow/lite/micro/%s.cc" % n for n in TFLM_MICRO]
        srcs.append("tensorflow/lite/micro/esp/micro_time.cc")
        kdir = os.path.join(src, "tensorflow/lite/micro/kernels")
        for fn in sorted(os.listdir(kdir)):
            if fn.endswith((".cc", ".c")) and os.path.splitext(fn)[0] not in ESP_NN_REPLACED:
                srcs.append("tensorflow/lite/micro/kernels/" + fn)
        for n in ESP_NN_REPLACED:
            srcs.append("tensorflow/lite/micro/kernels/esp_nn/%s.cc" % n)
        bdir = os.path.join(src, "tensorflow/lite/micro/tflite_bridge")
        for fn in sorted(os.listdir(bdir)):
            if fn.endswith((".cc", ".c")):
                srcs.append("tensorflow/lite/micro/tflite_bridge/" + fn)
        srcs += TFLM_EXPLICIT
        for rel in srcs:
            copy_file(src, rel, d)

        # 4) -O2 riêng cho depthwise_conv.cc của esp_nn (né lỗi trình biên dịch khi dùng -O3)
        p = os.path.join(d, "tensorflow/lite/micro/kernels/esp_nn/depthwise_conv.cc")
        with open(p, encoding="utf-8") as f:
            body = f.read()
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(O2_PRAGMA + body)

        with open(os.path.join(dst, "README.md"), "w", encoding="utf-8") as f:
            f.write("Vendor tu https://github.com/espressif/esp-tflite-micro @ %s (Apache-2.0)\n"
                    "Tao boi tools/setup_libs.py - khong sua tay.\n" % TFLM_COMMIT)
    print("[TFLiteMicro] xong: %d file nguon .cc/.c" % len(srcs))


def setup_esp_nn(force):
    dst = os.path.join(LIB, "esp-nn")
    if os.path.isdir(dst):
        if not force:
            print("[esp-nn] da co, bo qua (dung --force de dung lai)")
            return
        shutil.rmtree(dst)
    print("[esp-nn] dang dung...")
    with tempfile.TemporaryDirectory() as tmp:
        src = extract_root(download(NN_URL, "esp-nn " + NN_TAG), tmp)
        # header: include/ + mọi header trong src/ (trừ riscv)
        copy_tree_filtered(os.path.join(src, "include"), os.path.join(dst, "include"),
                           lambda rel: rel.endswith(".h"))
        copy_tree_filtered(os.path.join(src, "src"), os.path.join(dst, "src"),
                           lambda rel: rel.endswith(".h") and "riscv" not in rel)
        # source: đúng danh sách của ESP32-S3 (ANSI C + tối ưu assembly)
        for rel in ESP_NN_C + ESP_NN_S3:
            copy_file(src, rel, dst)
        with open(os.path.join(dst, "library.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write(ESP_NN_LIBRARY_JSON)
        shutil.copy2(os.path.join(src, "LICENSE"), os.path.join(dst, "LICENSE"))
    print("[esp-nn] xong: %d file nguon" % (len(ESP_NN_C) + len(ESP_NN_S3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="xoa va dung lai thu vien")
    args = ap.parse_args()
    os.makedirs(LIB, exist_ok=True)
    try:
        setup_tflm(args.force)
        setup_esp_nn(args.force)
    except Exception as e:  # noqa
        print("LOI: %s\nKiem tra ket noi internet roi chay lai (co the them --force)." % e)
        sys.exit(1)
    print("\nXONG. Bay gio dat include/model_data.h roi build: pio run")


if __name__ == "__main__":
    main()
