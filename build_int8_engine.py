#!/usr/bin/env python3
"""
YOLOv8m INT8 캘리브레이션 캐시 생성 및 엔진 빌드

[실행 순서]
  # 1단계: GPU·DLA 캘리브레이션 캐시 각각 생성 + GPU INT8 엔진 빌드
  python3 build_int8_engine.py --dla-core -1

  # 2단계: DLA 엔진 빌드 (DLA 캐시 재사용)
  python3 build_int8_engine.py --dla-core 0   → yolov8m_640_b1_dla0_int8.engine
  python3 build_int8_engine.py --dla-core 1   → yolov8m_640_b1_dla1_int8.engine

[캘리브레이션 분리 전략]
  GPU 캐시 (gpu_int8_calib.cache):
    - GPU 컨텍스트에서 MinMax 캘리브레이션
    - TensorRT가 per-channel 스케일로 변환하여 사용

  DLA 캐시 (dla_int8_calib.cache):
    - DLA 컨텍스트(default_device_type=DLA, GPU_FALLBACK)에서 캘리브레이션
    - DLA 서브그래프 경계에서 per-tensor 스케일 직접 계산
    - TRT Python 바인딩 DLA 직렬화 segfault → 서브프로세스로 우회
      (write_calibration_cache는 직렬화 전에 호출되므로 캐시는 안전하게 저장됨)
"""

import os
import re
import sys
import glob
import pickle
import tempfile
import argparse
import subprocess
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ── 고정 상수 ────────────────────────────────────────────────────────────────
MAX_CALIB_IMAGES    = 5000
CALIB_IMG_DIR       = "/home/nvidia/workspace/val2017"
WORKSPACE_GB        = 4      # 엔진 빌드용
CALIB_WORKSPACE_GB  = 2      # 캘리브레이션 전용 (통합 메모리 여유 확보)
BUILDER_OPT_LEVEL   = 5
AVG_TIMING_ITERS    = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8m INT8 캘리브레이션 캐시 생성 및 엔진 빌드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "실행 순서:\n"
            "  python3 build_int8_engine.py --dla-core -1  # 캐시 생성 + GPU 엔진\n"
            "  python3 build_int8_engine.py --dla-core 0   # DLA0 엔진\n"
            "  python3 build_int8_engine.py --dla-core 1   # DLA1 엔진\n"
        ),
    )
    parser.add_argument("--model",       default="yolov8m")
    parser.add_argument("--input-size",  type=int, default=640, metavar="N")
    parser.add_argument("--batch-size",  type=int, default=1, metavar="N",
                        help="TensorRT 엔진 배치 크기 (기본: 1, GPU 사용률 향상: 4~8 권장)")
    parser.add_argument("--dla-core",    type=int, default=-1, choices=[-1, 0, 1],
                        help="-1: GPU 엔진 빌드 + 캐시 생성, 0/1: DLA 엔진 빌드")
    parser.add_argument("--force-dla-calib", action="store_true",
                        help="DLA 캘리브레이션 캐시를 강제 재생성 (기존 파일 덮어씀)")
    parser.add_argument("--skip-gpu-engine", action="store_true",
                        help="GPU 엔진 빌드 생략 (캐시 생성만 수행)")

    # ── 서브프로세스 전용 숨은 인수 (DLA 캘리브레이션 워커) ─────────────────
    parser.add_argument("--_dla-worker",         action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-onnx",        default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_worker-cache",       default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_worker-paths-pkl",   default=None,        help=argparse.SUPPRESS)
    parser.add_argument("--_worker-dla-core",    type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-batch-size",  type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-input-size",  type=int, default=640, help=argparse.SUPPRESS)

    return parser.parse_args()


# ── 캘리브레이션 ONNX 자동 탐색 ──────────────────────────────────────────────
def find_calib_onnx(model, input_size):
    pattern = f"{model}_{input_size}_b*_calib_split.onnx"
    matches = glob.glob(pattern)
    if not matches:
        raise RuntimeError(f"캘리브레이션 ONNX 없음 (패턴: {pattern})")
    if len(matches) > 1:
        raise RuntimeError(f"캘리브레이션 ONNX 여러 개 발견: {matches}")
    path = matches[0]
    m = re.search(r"_b(\d+)_calib_split\.onnx$", path)
    if not m:
        raise RuntimeError(f"파일명에서 배치 크기 파싱 불가: {path}")
    return path, int(m.group(1))


# ── 전처리 ──────────────────────────────────────────────────────────────────
def preprocess(img_path, input_h, input_w):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    r = min(input_h / h, input_w / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (input_h - nh) // 2, (input_w - nw) // 2
    img = cv2.copyMakeBorder(img, top, input_h - nh - top, left, input_w - nw - left,
                              cv2.BORDER_CONSTANT, value=114)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


# ── INT8 캘리브레이터 ────────────────────────────────────────────────────────
class YOLOInt8Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, image_paths, batch_size, cache_file, input_h, input_w):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size  = batch_size
        self.cache_file  = cache_file
        self.input_h     = input_h
        self.input_w     = input_w
        self.cursor      = 0
        self.n_batches   = len(image_paths) // batch_size

        nbytes = batch_size * 3 * input_h * input_w * 4
        self._dev_mem = cuda.mem_alloc(nbytes)
        print(f"캘리브레이터: {len(image_paths)}장 / 배치={batch_size} → {self.n_batches}배치")

    def __del__(self):
        if hasattr(self, "_dev_mem"):
            self._dev_mem.free()

    def get_batch_size(self): return self.batch_size

    def get_batch(self, names):
        if self.cursor + self.batch_size > len(self.image_paths):
            return None
        batch_imgs = [preprocess(p, self.input_h, self.input_w)
                      for p in self.image_paths[self.cursor:self.cursor + self.batch_size]]
        batch_imgs = [b for b in batch_imgs if b is not None]
        if len(batch_imgs) < self.batch_size:
            return None
        cuda.memcpy_htod(self._dev_mem,
                         np.ascontiguousarray(np.stack(batch_imgs), dtype=np.float32))
        batch_idx = self.cursor // self.batch_size + 1
        print(f"  캘리브레이션 [{batch_idx:3d}/{self.n_batches}] "
              f"({self.cursor + self.batch_size}/{len(self.image_paths)}장)")
        self.cursor += self.batch_size
        return [int(self._dev_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"캘리브레이션 캐시 로드: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"캘리브레이션 캐시 저장: {self.cache_file}")


# ── 캘리브레이션 캐시 헤더 패치 ──────────────────────────────────────────────
# MinMaxCalibrator → EntropyCalibration2 헤더 패치 (trtexec 호환)
# 수치는 그대로 유지, 헤더 형식만 변경
def patch_calib_cache_for_trtexec(cache_file):
    with open(cache_file, "r") as f:
        lines = f.readlines()
    if lines and "MinMaxCalibration" in lines[0]:
        lines[0] = lines[0].replace("MinMaxCalibration", "EntropyCalibration2")
        with open(cache_file, "w") as f:
            f.writelines(lines)
        print(f"  캐시 헤더 패치: MinMaxCalibration → EntropyCalibration2")


# ── GPU 캘리브레이션 캐시 생성 ───────────────────────────────────────────────
def generate_gpu_calib_cache(image_paths, calib_onnx_path, calib_cache,
                              input_size, calib_batch_size):
    """GPU 컨텍스트에서 MinMax 캘리브레이션 캐시를 생성한다."""
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"[GPU 캘리브레이션] ONNX 파싱: {calib_onnx_path}")
    with open(calib_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX 파싱 실패")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, CALIB_WORKSPACE_GB * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.builder_optimization_level = 0   # 캐시만 생성, 엔진 최적화 불필요

    calibrator = YOLOInt8Calibrator(image_paths, calib_batch_size, calib_cache,
                                    input_size, input_size)
    config.int8_calibrator = calibrator

    print(f"  GPU 컨텍스트 캘리브레이션 실행 중...")
    builder.build_serialized_network(network, config)

    if not os.path.exists(calib_cache):
        raise RuntimeError(
            f"GPU 캘리브레이션 실패 — 캐시 파일이 생성되지 않음: {calib_cache}\n"
            f"GPU 메모리 부족일 경우 다른 프로세스를 종료 후 재시도하세요."
        )
    print(f"  완료 → {calib_cache}")
    patch_calib_cache_for_trtexec(calib_cache)


# ── DLA 캘리브레이션 워커 (서브프로세스에서 실행) ────────────────────────────
def _run_dla_calib_worker(args):
    """
    DLA 컨텍스트(default_device_type=DLA, GPU_FALLBACK)에서 캘리브레이션을 수행.
    서브프로세스에서 호출되므로 DLA 직렬화 segfault가 발생해도 메인 프로세스에 영향 없음.
    write_calibration_cache는 직렬화 이전에 호출되므로 캐시 파일은 안전하게 저장된다.
    """
    with open(args._worker_paths_pkl, "rb") as f:
        image_paths = pickle.load(f)

    calib_onnx_path = args._worker_onnx
    calib_cache     = args._worker_cache
    input_size      = args._worker_input_size
    calib_batch_size= args._worker_batch_size
    dla_core        = args._worker_dla_core

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"[DLA 캘리브레이션 워커] ONNX 파싱: {calib_onnx_path}")
    with open(calib_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX 파싱 실패")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, CALIB_WORKSPACE_GB * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    # DLA 컨텍스트 설정: 레이어를 DLA 서브그래프로 묶어 per-tensor 스케일 계산
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = dla_core
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.builder_optimization_level = 0   # 최소 빌드 (캐시 생성 후 가능한 빨리 종료)

    # DLA는 IInt8EntropyCalibrator2 타입만 허용 (MinMaxCalibrator는 DLA 검증 실패)
    calibrator = DLACalibrator(image_paths, calib_batch_size, calib_cache,
                               input_size, input_size)
    config.int8_calibrator = calibrator

    print(f"  DLA Core {dla_core} 컨텍스트 캘리브레이션 실행 중...")
    # write_calibration_cache 호출 후 DLA 직렬화에서 segfault 발생 가능
    # → 서브프로세스이므로 메인 프로세스에 영향 없음
    builder.build_serialized_network(network, config)

    if os.path.exists(calib_cache):
        print(f"  완료 → {calib_cache}")
    else:
        print(f"  경고: DLA 캘리브레이션 캐시 미생성 (DLA 빌드 실패)")


# ── DLA 캘리브레이션 캐시 생성 (서브프로세스 실행) ───────────────────────────
def generate_dla_calib_cache(image_paths, calib_onnx_path, calib_cache,
                              input_size, calib_batch_size, dla_core=0):
    """
    DLA 전용 캘리브레이션 캐시를 서브프로세스에서 생성한다.

    GPU 캐시와의 차이:
      GPU: GPU 컨텍스트 → per-channel 스케일 → TRT가 DLA 적용 시 per-tensor로 변환
      DLA: DLA 컨텍스트 → 레이어가 DLA 서브그래프로 묶인 상태에서 per-tensor 스케일 직접 계산
           → DLA 실제 실행 경로와 일치하는 양자화 스케일
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(image_paths, tmp)
        tmp_paths_pkl = tmp.name

    try:
        cmd = [
            sys.executable, __file__,
            "--_dla-worker",
            "--_worker-onnx",       calib_onnx_path,
            "--_worker-cache",      calib_cache,
            "--_worker-paths-pkl",  tmp_paths_pkl,
            "--_worker-dla-core",   str(dla_core),
            "--_worker-batch-size", str(calib_batch_size),
            "--_worker-input-size", str(input_size),
        ]
        print(f"[DLA 캘리브레이션] DLA Core {dla_core} 컨텍스트 서브프로세스 실행...")
        print(f"  DLA 직렬화 segfault 발생 시 캐시는 이미 저장되어 있으므로 정상")
        proc = subprocess.run(cmd, timeout=300)   # 5분 이내 완료 기대, 초과 시 강제 종료

        if os.path.exists(calib_cache):
            patch_calib_cache_for_trtexec(calib_cache)
            size_kb = os.path.getsize(calib_cache) / 1024
            print(f"  DLA 캘리브레이션 캐시 생성 완료: {calib_cache}  ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  경고: DLA 캘리브레이션 캐시 생성 실패 (exit={proc.returncode})")
            print(f"  GPU 캐시를 DLA 캐시로 복사하여 계속 진행합니다.")
            return False
    finally:
        if os.path.exists(tmp_paths_pkl):
            os.unlink(tmp_paths_pkl)


# ── DLA 전용 캘리브레이터 ────────────────────────────────────────────────────
# DLA 빌더는 IInt8EntropyCalibrator2 타입만 허용한다.
# MinMax 방식이 아닌 TRT 내장 엔트로피 기반 스케일 계산을 사용하며,
# DLA 서브그래프 경계에서 per-tensor 스케일이 결정된다.
class DLACalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, batch_size, cache_file, input_h, input_w):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size  = batch_size
        self.cache_file  = cache_file
        self.input_h     = input_h
        self.input_w     = input_w
        self.cursor      = 0
        self.n_batches   = len(image_paths) // batch_size

        nbytes = batch_size * 3 * input_h * input_w * 4
        self._dev_mem = cuda.mem_alloc(nbytes)
        print(f"DLA 캘리브레이터: {len(image_paths)}장 / 배치={batch_size} → {self.n_batches}배치")

    def __del__(self):
        if hasattr(self, "_dev_mem"):
            self._dev_mem.free()

    def get_batch_size(self): return self.batch_size

    def get_batch(self, names):
        if self.cursor + self.batch_size > len(self.image_paths):
            return None
        batch_imgs = [preprocess(p, self.input_h, self.input_w)
                      for p in self.image_paths[self.cursor:self.cursor + self.batch_size]]
        batch_imgs = [b for b in batch_imgs if b is not None]
        if len(batch_imgs) < self.batch_size:
            return None
        cuda.memcpy_htod(self._dev_mem,
                         np.ascontiguousarray(np.stack(batch_imgs), dtype=np.float32))
        batch_idx = self.cursor // self.batch_size + 1
        print(f"  DLA 캘리브레이션 [{batch_idx:3d}/{self.n_batches}] "
              f"({self.cursor + self.batch_size}/{len(self.image_paths)}장)")
        self.cursor += self.batch_size
        return [int(self._dev_mem)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"DLA 캘리브레이션 캐시 로드: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"DLA 캘리브레이션 캐시 저장: {self.cache_file}")


# ── GPU 캐시 리더 ─────────────────────────────────────────────────────────────
class _CacheReader(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size, input_h, input_w):
        super().__init__()
        self._cache_file = cache_file
        self._batch_size = batch_size
        nbytes = batch_size * 3 * input_h * input_w * 4
        self._dev_mem = cuda.mem_alloc(nbytes)

    def __del__(self):
        if hasattr(self, "_dev_mem"):
            self._dev_mem.free()

    def get_batch_size(self):    return self._batch_size
    def get_batch(self, names):  return None

    def read_calibration_cache(self):
        with open(self._cache_file, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache): pass


# ── GPU INT8 엔진 빌드 ────────────────────────────────────────────────────────
def build_engine_gpu(engine_path, onnx_path, calib_cache, input_size):
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"[GPU] ONNX 파싱: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX 파싱 실패")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = BUILDER_OPT_LEVEL
    config.int8_calibrator = _CacheReader(calib_cache, batch_size, input_size, input_size)

    print(f"  GPU INT8 엔진 빌드 중 (최적화 레벨={BUILDER_OPT_LEVEL})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("GPU INT8 엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  완료: {engine_path}  ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


# ── DLA 엔진 빌드 (trtexec) ───────────────────────────────────────────────────
def build_engine_dla(engine_path, onnx_path, dla_core, calib_cache, timing_cache):
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--int8", "--fp16",
        f"--useDLACore={dla_core}",
        "--allowGPUFallback",
        f"--calib={calib_cache}",
        "--sparsity=enable",
        f"--builderOptimizationLevel={BUILDER_OPT_LEVEL}",
        f"--timingCacheFile={timing_cache}",
        f"--memPoolSize=workspace:{WORKSPACE_GB * 1024}MiB",
        f"--avgTiming={AVG_TIMING_ITERS}",
    ]
    print(f"[DLA Core {dla_core}] 빌드 중...")
    print(f"  {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"trtexec 실패 (exit={ret.returncode})")
    print(f"  완료: {engine_path}  ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 서브프로세스 워커 모드 ─────────────────────────────────────────────
    if args._dla_worker:
        _run_dla_calib_worker(args)
        return

    model      = args.model
    input_size = args.input_size
    dla_core   = args.dla_core
    batch_size = args.batch_size

    calib_onnx, calib_batch_size = find_calib_onnx(model, input_size)

    onnx_path       = f"{model}_{input_size}_split.onnx"
    gpu_calib_cache = f"{model}_{input_size}_gpu_int8_calib.cache"
    dla_calib_cache = f"{model}_{input_size}_dla_int8_calib.cache"

    if dla_core >= 0:
        # ── DLA 엔진 빌드 ─────────────────────────────────────────────────
        engine_path  = f"{model}_{input_size}_b{batch_size}_dla{dla_core}_int8.engine"
        timing_cache = f"{model}_{input_size}_dla{dla_core}_timing.cache"

        print(f"=== YOLOv8m DLA{dla_core} INT8 엔진 빌드 ===")
        print(f"ONNX   : {onnx_path}")
        print(f"엔진   : {engine_path}")
        print(f"캐시   : {dla_calib_cache}\n")

        if not os.path.exists(dla_calib_cache):
            raise RuntimeError(
                f"DLA 캘리브레이션 캐시 없음 → 먼저 --dla-core -1 로 실행: {dla_calib_cache}"
            )
        build_engine_dla(engine_path, onnx_path, dla_core, dla_calib_cache, timing_cache)
        print(f"\n타이밍 캐시: {timing_cache}")

    else:
        # ── 캘리브레이션 캐시 생성 + GPU 엔진 빌드 ────────────────────────
        print(f"=== YOLOv8m INT8 캘리브레이션 캐시 생성 ===")
        print(f"캘리브 ONNX  : {calib_onnx}  (batch={calib_batch_size})")
        print(f"캘리브 이미지 : {CALIB_IMG_DIR}  최대 {MAX_CALIB_IMAGES}장")
        print(f"GPU 캐시     : {gpu_calib_cache}")
        print(f"DLA 캐시     : {dla_calib_cache}\n")

        all_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            all_files.extend(glob.glob(os.path.join(CALIB_IMG_DIR, f"*{ext}")))
        if not all_files:
            raise RuntimeError(f"이미지를 찾을 수 없음: {CALIB_IMG_DIR}")
        np.random.shuffle(all_files)
        all_files = all_files[:MAX_CALIB_IMAGES]
        print(f"캘리브레이션 이미지: {len(all_files)}장 선택\n")

        # GPU 캐시 생성
        legacy_cache = f"{model}_{input_size}_int8_calib.cache"   # 기존 공유 캐시
        if os.path.exists(gpu_calib_cache):
            print(f"GPU 캐시 이미 존재, 재사용: {gpu_calib_cache}")
        elif os.path.exists(legacy_cache):
            # 기존 캐시는 GPU 컨텍스트에서 생성된 것이므로 GPU 캐시로 재사용
            import shutil
            shutil.copy2(legacy_cache, gpu_calib_cache)
            print(f"기존 캐시 → GPU 캐시로 복사: {legacy_cache} → {gpu_calib_cache}")
        else:
            print("─── [1/2] GPU 캘리브레이션 캐시 ───────────────────────────────")
            generate_gpu_calib_cache(all_files, calib_onnx, gpu_calib_cache,
                                     input_size, calib_batch_size)

        # DLA 캐시 생성 (서브프로세스)
        if os.path.exists(dla_calib_cache) and not args.force_dla_calib:
            print(f"DLA 캐시 이미 존재, 재사용: {dla_calib_cache}")
        else:
            print("\n─── [2/2] DLA 캘리브레이션 캐시 ───────────────────────────────")
            ok = generate_dla_calib_cache(all_files, calib_onnx, dla_calib_cache,
                                          input_size, calib_batch_size, dla_core=0)
            if not ok:
                import shutil
                shutil.copy2(gpu_calib_cache, dla_calib_cache)
                print(f"  폴백: GPU 캐시를 DLA 캐시로 복사 ({dla_calib_cache})")

        # GPU INT8 엔진 빌드
        gpu_engine_path = f"{model}_{input_size}_b{batch_size}_gpu_int8.engine"
        if args.skip_gpu_engine:
            print(f"\nGPU 엔진 빌드 생략 (--skip-gpu-engine)")
        else:
            print(f"\n=== YOLOv8m GPU INT8 엔진 빌드 ===")
            print(f"ONNX  : {onnx_path}")
            print(f"엔진  : {gpu_engine_path}")
            print(f"캐시  : {gpu_calib_cache}\n")
            build_engine_gpu(gpu_engine_path, onnx_path, gpu_calib_cache, input_size)

        print(f"\n완료:")
        print(f"  GPU 캐시: {gpu_calib_cache}")
        print(f"  DLA 캐시: {dla_calib_cache}")
        print(f"  GPU 엔진: {gpu_engine_path}")
        print("다음 단계:")
        print(f"  python3 build_int8_engine.py --dla-core 0")
        print(f"  python3 build_int8_engine.py --dla-core 1")


if __name__ == "__main__":
    main()
