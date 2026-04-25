#!/usr/bin/env python3
"""
yolo11m.onnx → TensorRT INT8 엔진 변환

INT8 캘리브레이션 데이터는 video_h264.mp4 에서 균등 추출한 프레임을 사용.
캘리브레이션 캐시는 yolo11m_int8_calib.cache 에 저장되어 재실행 시 재사용됨.
빌더 타이밍 캐시는 yolo11m_timing.cache 에 저장되어 재빌드 시 커널 선택을 재사용.

사용법:
  python3 build_int8_engine.py
"""

import os
import ctypes
import numpy as np
import cv2
import tensorrt as trt

# ── 설정 ────────────────────────────────────────────────────────────────────────
ONNX_PATH       = "yolo11m.onnx"
VIDEO_PATH      = "video_h264.mp4"
CALIB_CACHE     = "yolo11m_int8_calib.cache"

BATCH_SIZE      = 4
INPUT_H         = 640
INPUT_W         = 640
N_CALIB_BATCHES = 25          # 25 배치 × 4 = 100 프레임
WORKSPACE_GB    = 4

# ── DLA 설정 ─────────────────────────────────────────────────────────────────
# -1 = GPU 전용,  0 또는 1 = DLA 코어 번호 (Jetson Orin: 코어 2개)
DLA_CORE = 0

# DLA_CORE 값에 따라 출력 파일명 자동 결정
ENGINE_PATH  = (f"yolo11m_b4_dla{DLA_CORE}_int8.engine"
                if DLA_CORE >= 0 else "yolo11m_b4_int8.engine")
TIMING_CACHE = (f"yolo11m_dla{DLA_CORE}_timing.cache"
                if DLA_CORE >= 0 else "yolo11m_timing.cache")

# ── 빌더 성능 옵션 ────────────────────────────────────────────────────────────
BUILDER_OPT_LEVEL   = 5   # 0~5, 기본=3. 높을수록 빌드 시간↑ 추론 성능↑
AVG_TIMING_ITERS    = 12  # 커널 선택 시 평균 측정 횟수 (기본=8)
MIN_TIMING_ITERS    = 2   # 커널 선택 최소 측정 횟수 (기본=1)
MAX_AUX_STREAMS     = 4   # GPU fallback 레이어 병렬 실행용 보조 스트림

# ── CUDA ctypes 래퍼 ─────────────────────────────────────────────────────────
libcudart = ctypes.CDLL("libcudart.so", use_errno=True)
libcudart.cudaMalloc.restype    = ctypes.c_int
libcudart.cudaFree.restype      = ctypes.c_int
libcudart.cudaMemcpy.restype    = ctypes.c_int

MEMCPY_H2D = 1  # cudaMemcpyHostToDevice

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    ret = libcudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    if ret != 0:
        raise RuntimeError(f"cudaMalloc 실패: error={ret}, size={nbytes}")
    return ptr.value

def cuda_free(ptr):
    libcudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_memcpy_h2d(dst_ptr, src_array):
    src = np.ascontiguousarray(src_array, dtype=np.float32)
    ret = libcudart.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.nbytes),
        ctypes.c_int(MEMCPY_H2D),
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy 실패: error={ret}")


# ── 전처리 ──────────────────────────────────────────────────────────────────────
def letterbox(img, size=(640, 640), pad_color=114):
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top  = (size[0] - nh) // 2
    left = (size[1] - nw) // 2
    bottom = size[0] - nh - top
    right  = size[1] - nw - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=pad_color)

def preprocess(bgr):
    img = letterbox(bgr, (INPUT_H, INPUT_W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)   # CHW float32


# ── 캘리브레이션 프레임 추출 ────────────────────────────────────────────────────
def extract_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"비디오를 열 수 없음: {video_path}")
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(preprocess(frame))
    cap.release()
    print(f"  비디오 {total}프레임 중 {len(frames)}프레임 추출 완료")
    return frames


# ── INT8 캘리브레이터 ────────────────────────────────────────────────────────────
class YOLOInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, frames, batch_size, cache_file):
        super().__init__()
        self.batch_size  = batch_size
        self.cache_file  = cache_file
        self.frames      = frames
        self.cursor      = 0
        self.n_batches   = len(frames) // batch_size

        nbytes = batch_size * 3 * INPUT_H * INPUT_W * 4   # float32
        self._dev_ptr = cuda_malloc(nbytes)
        self._nbytes  = nbytes
        print(f"캘리브레이터 초기화: {self.n_batches} 배치 × {batch_size} 이미지")

    def __del__(self):
        if hasattr(self, "_dev_ptr") and self._dev_ptr:
            cuda_free(self._dev_ptr)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        end = self.cursor + self.batch_size
        if end > len(self.frames):
            return None
        batch = np.stack(self.frames[self.cursor:end])
        cuda_memcpy_h2d(self._dev_ptr, batch)
        batch_idx = self.cursor // self.batch_size + 1
        print(f"  캘리브레이션 [{batch_idx:2d}/{self.n_batches}] "
              f"프레임 {self.cursor}~{end-1}")
        self.cursor = end
        return [self._dev_ptr]

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


# ── DLA 엔진 빌드 (trtexec) ──────────────────────────────────────────────────────
# TRT 8.6.2 Python 바인딩은 DLA 엔진 직렬화 시 segfault 버그가 있음.
# C++ 바이너리인 trtexec는 동일 조건에서 정상 동작하므로 DLA 빌드에 사용.
def build_engine_dla(onnx_path, engine_path):
    import subprocess
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--int8",
        "--fp16",
        f"--useDLACore={DLA_CORE}",
        "--allowGPUFallback",
        f"--calib={CALIB_CACHE}",
        "--sparsity=enable",
        f"--builderOptimizationLevel={BUILDER_OPT_LEVEL}",
        f"--timingCacheFile={TIMING_CACHE}",
        f"--memPoolSize=workspace:{WORKSPACE_GB * 1024}MiB",
        f"--avgTiming={AVG_TIMING_ITERS}",
    ]
    print(f"엔진 빌드 중 (DLA core {DLA_CORE} + GPU fallback, INT8+FP16)...")
    print(f"  명령: {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"trtexec 실패 (exit={ret.returncode})")
    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"\n완료: {engine_path} ({size_mb:.1f} MB)")


# ── GPU 엔진 빌드 (Python TRT API) ───────────────────────────────────────────────
def build_engine_gpu(onnx_path, engine_path, calibrator):
    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"ONNX 파싱: {onnx_path}")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  오류: {parser.get_error(i)}")
        raise RuntimeError("ONNX 파싱 실패")
    print(f"  입력: {network.get_input(0).shape}")
    print(f"  출력: {network.get_output(0).shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB * (1 << 30)
    )
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.int8_calibrator = calibrator
    config.builder_optimization_level = BUILDER_OPT_LEVEL
    config.avg_timing_iterations = AVG_TIMING_ITERS
    config.min_timing_iterations = MIN_TIMING_ITERS
    config.max_aux_streams = MAX_AUX_STREAMS
    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT) |
        1 << int(trt.TacticSource.CUDNN) |
        1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS) |
        1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
    )

    timing_cache_data = b""
    if os.path.exists(TIMING_CACHE):
        print(f"타이밍 캐시 로드: {TIMING_CACHE}")
        with open(TIMING_CACHE, "rb") as f:
            timing_cache_data = f.read()
    timing_cache = config.create_timing_cache(timing_cache_data)
    config.set_timing_cache(timing_cache, ignore_mismatch=True)

    print(f"엔진 빌드 중 (GPU, INT8+FP16, opt_level={BUILDER_OPT_LEVEL}, workspace={WORKSPACE_GB}GB)...")
    print("  처음 빌드 시 수십 분이 소요됩니다.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    with timing_cache.serialize() as buf:
        with open(TIMING_CACHE, "wb") as f:
            f.write(bytes(buf))
    print(f"타이밍 캐시 저장: {TIMING_CACHE}")

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"\n완료: {engine_path} ({size_mb:.1f} MB)")


# ── 메인 ────────────────────────────────────────────────────────────────────────
def main():
    n_total = N_CALIB_BATCHES * BATCH_SIZE
    mode = f"DLA core {DLA_CORE} + GPU fallback" if DLA_CORE >= 0 else "GPU"
    print(f"=== YOLO11m INT8 엔진 빌드 ({mode}) ===")
    print(f"ONNX   : {ONNX_PATH}")
    print(f"Engine : {ENGINE_PATH}")
    print(f"캘리브레이션: {n_total}프레임 ({N_CALIB_BATCHES}배치 × {BATCH_SIZE})")
    print()

    print(f"[1/3] 캘리브레이션 프레임 추출 ({VIDEO_PATH})")
    frames = extract_frames(VIDEO_PATH, n_total)
    if len(frames) < BATCH_SIZE:
        raise RuntimeError("캘리브레이션 프레임이 부족합니다.")

    print(f"\n[3/3] TensorRT 엔진 빌드")
    if DLA_CORE >= 0:
        # DLA: trtexec 사용 (Python 바인딩 segfault 우회)
        # 캘리브레이션 캐시가 없으면 먼저 GPU 빌드로 생성 필요
        if not os.path.exists(CALIB_CACHE):
            print(f"  캘리브레이션 캐시 없음 → GPU 빌드로 먼저 생성하세요 (DLA_CORE=-1)")
            raise RuntimeError(f"캘리브레이션 캐시 필요: {CALIB_CACHE}")
        build_engine_dla(ONNX_PATH, ENGINE_PATH)
    else:
        print(f"\n[2/3] 캘리브레이터 준비")
        calibrator = YOLOInt8Calibrator(frames, BATCH_SIZE, CALIB_CACHE)
        build_engine_gpu(ONNX_PATH, ENGINE_PATH, calibrator)

    print(f"\n캘리브레이션 캐시: {CALIB_CACHE}")
    print(f"타이밍 캐시     : {TIMING_CACHE} (재빌드 시 커널 선택 재사용)")


if __name__ == "__main__":
    main()
