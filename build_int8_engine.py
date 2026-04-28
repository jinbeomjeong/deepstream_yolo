#!/usr/bin/env python3
"""
YOLO11m INT8 캘리브레이션 캐시 생성 및 DLA 엔진 빌드

[실행 순서]
  # 1단계: 캘리브레이션 캐시 생성 (GPU 사용, 엔진 파일 생성 없음)
  DLA_CORE=-1 python3 build_int8_engine.py

  # 2단계: DLA 엔진 빌드 (캐시 재사용)
  DLA_CORE=0 python3 build_int8_engine.py   → yolo11m_b2_dla0_int8.engine
  DLA_CORE=1 python3 build_int8_engine.py   → yolo11m_b2_dla1_int8.engine

[캘리브레이션 전략]
  캘리브레이션 전용 ONNX (batch=CALIB_BATCH_SIZE=100) 로 빠르게 calib cache 생성.
  TRT get_batch 호출 수 = 2000 / 100 = 20회/pass  (batch=2 대비 50배 감소)

  왜 MinMaxCalibrator 인가:
    IInt8EntropyCalibrator2 는 희소한 고신뢰도 앵커를 KL 최적화로 클리핑해
    sigmoid 출력 스케일을 과소 추정 → 탐지 확률이 0으로 수렴하는 버그 발생.
    MinMax 는 실제 최대값을 보존해 이 문제를 방지함.
"""

import os
import glob
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit   # CUDA 컨텍스트 자동 초기화

# ── 모델 / 입력 설정 ──────────────────────────────────────────────────────────────
# 환경변수로 주입:
#   MODEL=yolov8m INPUT_SIZE=512 DLA_CORE=0 python3 build_int8_engine.py
MODEL            = os.environ.get("MODEL",      "yolov8m")
INPUT_SIZE       = int(os.environ.get("INPUT_SIZE", "512"))

ONNX_PATH        = f"{MODEL}_{INPUT_SIZE}_split.onnx"       # boxes/classes 분리 출력
CALIB_CACHE      = f"{MODEL}_{INPUT_SIZE}_int8_calib.cache"

BATCH_SIZE       = 2                         # 엔진 배치 크기 (DLA 요건)
CALIB_BATCH_SIZE = 100                       # 캘리브레이션 배치 크기 (클수록 빠름)
CALIB_ONNX_PATH  = f"{MODEL}_{INPUT_SIZE}_b{CALIB_BATCH_SIZE}_calib_split.onnx"

INPUT_H          = INPUT_SIZE
INPUT_W          = INPUT_SIZE
MAX_CALIB_IMAGES = 2000
CALIB_IMG_DIR    = "/home/nvidia/workspace/train2017"
WORKSPACE_GB     = 4

BUILDER_OPT_LEVEL = 5
AVG_TIMING_ITERS  = 12

# ── DLA 설정 ─────────────────────────────────────────────────────────────────
# 환경변수로 주입: DLA_CORE=0 python3 build_int8_engine.py
DLA_CORE     = int(os.environ.get("DLA_CORE", "-1"))
ENGINE_PATH  = f"{MODEL}_{INPUT_SIZE}_b{BATCH_SIZE}_dla{DLA_CORE}_int8.engine"
TIMING_CACHE = f"{MODEL}_{INPUT_SIZE}_dla{DLA_CORE}_timing.cache"



# ── 전처리 ──────────────────────────────────────────────────────────────────────
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    r = min(INPUT_H / h, INPUT_W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (INPUT_H - nh) // 2, (INPUT_W - nw) // 2
    img = cv2.copyMakeBorder(img, top, INPUT_H - nh - top, left, INPUT_W - nw - left,
                              cv2.BORDER_CONSTANT, value=114)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)   # CHW float32


# ── INT8 캘리브레이터 (on-demand 로딩) ─────────────────────────────────────────
class YOLOInt8Calibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, image_paths, batch_size, cache_file):
        super().__init__()
        self.image_paths = image_paths
        self.batch_size  = batch_size
        self.cache_file  = cache_file
        self.cursor      = 0
        self.n_batches   = len(image_paths) // batch_size

        nbytes = batch_size * 3 * INPUT_H * INPUT_W * 4   # float32
        self._dev_mem = cuda.mem_alloc(nbytes)
        print(f"캘리브레이터: {len(image_paths)}장 / 배치={batch_size} → {self.n_batches}배치")

    def __del__(self):
        if hasattr(self, "_dev_mem"):
            self._dev_mem.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.cursor + self.batch_size > len(self.image_paths):
            return None

        batch_imgs = []
        for path in self.image_paths[self.cursor:self.cursor + self.batch_size]:
            img = preprocess(path)
            if img is not None:
                batch_imgs.append(img)

        if len(batch_imgs) < self.batch_size:
            return None

        cuda.memcpy_htod(self._dev_mem, np.ascontiguousarray(np.stack(batch_imgs), dtype=np.float32))
        batch_idx = self.cursor // self.batch_size + 1
        #if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == self.n_batches:
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



# ── 캘리브레이션 캐시 헤더 패치 ───────────────────────────────────────────────
# Python IInt8MinMaxCalibrator 는 "MinMaxCalibration" 헤더를 쓰지만,
# trtexec 는 "EntropyCalibration2" 헤더를 기대함.
# 헤더 불일치 시 trtexec 가 캐시를 무효화하고 빈 데이터로 재캘리브레이션 → CUDA crash.
# per-tensor 수치(MinMax 계산값)는 그대로 유지하므로 탐지 정확도 영향 없음.
def patch_calib_cache_for_trtexec(cache_file):
    with open(cache_file, "r") as f:
        lines = f.readlines()
    if lines and "MinMaxCalibration" in lines[0]:
        lines[0] = lines[0].replace("MinMaxCalibration", "EntropyCalibration2")
        with open(cache_file, "w") as f:
            f.writelines(lines)
        print(f"  캐시 헤더 패치: MinMaxCalibration → EntropyCalibration2 (trtexec 호환)")


# ── 캘리브레이션 캐시 생성 ─────────────────────────────────────────────────────
# CALIB_ONNX (batch=CALIB_BATCH_SIZE) 로 INT8 캘리브레이션 실행 → cache 저장, 엔진 버림
def generate_calib_cache(image_paths):
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"[캘리브레이션] ONNX 파싱: {CALIB_ONNX_PATH}")
    with open(CALIB_ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  파싱 오류: {parser.get_error(i)}")
            raise RuntimeError("캘리브레이션 ONNX 파싱 실패")
    print(f"  입력: {network.get_input(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30))
    config.set_flag(trt.BuilderFlag.INT8)
    # 캘리브레이션 후 버릴 엔진이므로 최적화 수준을 최소로 설정
    # (캘리브레이션 캐시 품질은 최적화 수준과 무관)
    config.builder_optimization_level = 0
    calibrator = YOLOInt8Calibrator(image_paths, CALIB_BATCH_SIZE, CALIB_CACHE)
    config.int8_calibrator = calibrator

    n_batches = len(image_paths) // CALIB_BATCH_SIZE
    print(f"  {len(image_paths)}장 / {CALIB_BATCH_SIZE}장/배치 = {n_batches}배치")
    print("  캘리브레이션 실행 중... (캐시 저장 후 버릴 엔진은 최소 빌드)")
    builder.build_serialized_network(network, config)
    print(f"  캘리브레이션 완료 → {CALIB_CACHE}")
    patch_calib_cache_for_trtexec(CALIB_CACHE)


# ── DLA 엔진 빌드 (trtexec) ───────────────────────────────────────────────────
# TRT 8.6.2 Python 바인딩은 DLA 직렬화 시 segfault 버그 → trtexec 우회
def build_engine_dla(engine_path):
    import subprocess
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    cmd = [
        trtexec,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={engine_path}",
        "--int8", "--fp16",
        f"--useDLACore={DLA_CORE}",
        "--allowGPUFallback",
        f"--calib={CALIB_CACHE}",
        "--sparsity=enable",
        f"--builderOptimizationLevel={BUILDER_OPT_LEVEL}",
        f"--timingCacheFile={TIMING_CACHE}",
        f"--memPoolSize=workspace:{WORKSPACE_GB * 1024}MiB",
        f"--avgTiming={AVG_TIMING_ITERS}",
    ]
    print(f"[DLA Core {DLA_CORE}] 빌드 중...")
    print(f"  {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"trtexec 실패 (exit={ret.returncode})")
    print(f"  완료: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


# ── 메인 ────────────────────────────────────────────────────────────────────────
def main():
    if DLA_CORE >= 0:
        # ── DLA 엔진 빌드 ─────────────────────────────────────────────────────
        print(f"=== YOLO11m DLA{DLA_CORE} INT8 엔진 빌드 ===")
        print(f"ONNX  : {ONNX_PATH}  (batch={BATCH_SIZE})")
        print(f"엔진  : {ENGINE_PATH}")
        print(f"캐시  : {CALIB_CACHE}")
        print()
        if not os.path.exists(CALIB_CACHE):
            raise RuntimeError(
                f"캘리브레이션 캐시 없음 → 먼저 DLA_CORE=-1 로 실행하세요: {CALIB_CACHE}")
        build_engine_dla(ENGINE_PATH)
        print(f"\n타이밍 캐시: {TIMING_CACHE}")

    else:
        # ── 캘리브레이션 캐시 생성 (엔진 파일 생성 없음) ────────────────────────
        print(f"=== YOLO11m INT8 캘리브레이션 캐시 생성 ===")
        print(f"캘리브 ONNX : {CALIB_ONNX_PATH}  (batch={CALIB_BATCH_SIZE})")
        print(f"캘리브 이미지: {CALIB_IMG_DIR}  최대 {MAX_CALIB_IMAGES}장")
        print(f"출력 캐시   : {CALIB_CACHE}")
        print()

        # 이미지 수집
        all_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            all_files.extend(glob.glob(os.path.join(CALIB_IMG_DIR, f"*{ext}")))
        if not all_files:
            raise RuntimeError(f"이미지를 찾을 수 없음: {CALIB_IMG_DIR}")
        np.random.shuffle(all_files)
        all_files = all_files[:MAX_CALIB_IMAGES]
        print(f"[1/2] 캘리브레이션 이미지: {len(all_files)}장 선택\n")

        # 캘리브레이션 ONNX 확인
        if not os.path.exists(CALIB_ONNX_PATH):
            raise RuntimeError(
                f"캘리브레이션 ONNX 없음 → 먼저 export_yolo11.py 를 실행하세요: {CALIB_ONNX_PATH}")
        print(f"[2/2] 캘리브레이션 ONNX: {CALIB_ONNX_PATH}\n")

        # 캘리브레이션 캐시 생성
        if os.path.exists(CALIB_CACHE):
            print(f"캘리브레이션 캐시 이미 존재, 재사용: {CALIB_CACHE}")
        else:
            generate_calib_cache(all_files)

        print(f"\n완료: {CALIB_CACHE}")
        print("다음 단계:")
        print(f"  DLA_CORE=0 python3 build_int8_engine.py")
        print(f"  DLA_CORE=1 python3 build_int8_engine.py")


if __name__ == "__main__":
    main()
