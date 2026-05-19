#!/usr/bin/env python3
"""
YOLOv8m Mixed Precision DLA+GPU Engine Builder
===============================================
[전략]
  Backbone + Neck  →  DLA Core N,  INT8   (속도 유지)
  Detection Head   →  GPU,         FP16   (bbox/cls 정밀도 보존)

[이유]
  INT8 양자화는 좌표 텐서(0~640)와 확률 텐서(0~1)의 스케일 격차를
  누적 오차로 확대함.  split output 구조로 1차 완화했지만,
  detection head 연산 자체의 정밀도 손실은 남아 있음.
  해당 레이어만 GPU FP16으로 실행하면 backbone 가속을 유지하면서
  bbox/cls 출력 품질을 보존할 수 있음.

[Detection Head 식별]
  YOLOv8m ONNX: /model.22/ 패턴 레이어 (Detect module)
  - cv2 브랜치 : bbox 좌표 회귀 (4 channels)
  - cv3 브랜치 : 클래스 확률    (80 channels)
  - dfl         : bbox 분포 디코딩

[빌드 방식]
  1. TRT Python API (GPU 모드, DLA 없음) 로 ONNX 파싱 → 레이어명 수집
  2. trtexec --layerDeviceTypes + --layerPrecisions 로 엔진 빌드
     (Python API DLA 직렬화 segfault 우회, 기존 전략 유지)

[실행 순서]
  # 캘리브레이션 캐시가 없으면 먼저 생성
  python3 build_int8_engine.py --dla-core -1

  # detection head 레이어 확인 (엔진 빌드 없이)
  python3 build_mixed_engine.py --inspect

  # DLA0, DLA1 혼합 정밀도 엔진 빌드
  python3 build_mixed_engine.py --dla-core 0
  python3 build_mixed_engine.py --dla-core 1
"""

import os
import sys
import subprocess
import argparse
import tensorrt as trt

WORK_DIR       = "/home/nvidia/workspace/deepstream_yolo"
BATCH_SIZE     = 2
WORKSPACE_GB   = 4
BUILDER_OPT    = 5
AVG_TIMING     = 12
TRTEXEC        = "/usr/src/tensorrt/bin/trtexec"
HEAD_PATTERN   = "/model.22/"   # YOLOv8 Detect module ONNX 노드 접두사


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="YOLOv8m Backbone(DLA INT8) + Detection Head(GPU FP16) 엔진 빌드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python3 build_mixed_engine.py --inspect\n"
            "  python3 build_mixed_engine.py --dla-core 0\n"
            "  python3 build_mixed_engine.py --dla-core 1\n"
        ),
    )
    p.add_argument("--model",      default="yolov8m",
                   help="모델 이름 (기본값: yolov8m)")
    p.add_argument("--input-size", type=int, default=640, metavar="N",
                   help="입력 해상도 (기본값: 640)")
    p.add_argument("--dla-core",   type=int, default=-1, choices=[-1, 0, 1],
                   help="-1: 검사 전용, 0/1: 해당 DLA 코어용 엔진 빌드 (기본값: -1)")
    p.add_argument("--inspect",    action="store_true",
                   help="detection head 레이어 목록만 출력하고 종료")
    p.add_argument("--head-pattern", default=HEAD_PATTERN,
                   help=f"detection head 레이어 필터 패턴 (기본값: {HEAD_PATTERN!r})")
    return p.parse_args()


# ── Detection Head 레이어 식별 ───────────────────────────────────────────────
def identify_head_layers(onnx_path: str, pattern: str) -> list[str]:
    """
    TRT Python API (GPU 전용, DLA 없음) 로 ONNX 파싱 후
    pattern 을 포함하는 TRT 레이어명 반환.

    GPU 모드로만 파싱하므로 DLA 직렬화 segfault 가 발생하지 않음.
    trtexec 가 내부적으로 사용하는 레이어명과 동일한 이름을 얻을 수 있음.
    """
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  파싱 오류: {parser.get_error(i)}", file=sys.stderr)
            raise RuntimeError("ONNX 파싱 실패")

    all_names  = [network.get_layer(i).name for i in range(network.num_layers)]
    head_names = [n for n in all_names if pattern in n]

    print(f"  전체 TRT 레이어  : {len(all_names)}개")
    print(f"  Detection head   : {len(head_names)}개  (패턴: {pattern!r})")

    if not head_names:
        print("\n  ⚠  패턴 불일치 — 실제 레이어명 상위 30개:")
        for name in all_names[:30]:
            print(f"      {name}")
        print("  --head-pattern 옵션으로 올바른 패턴을 지정하세요.")

    return head_names


# ── trtexec 혼합 정밀도 엔진 빌드 ────────────────────────────────────────────
def build_engine_mixed(
    onnx_path: str,
    engine_path: str,
    dla_core: int,
    calib_cache: str,
    timing_cache: str,
    head_layers: list[str],
) -> None:
    """
    trtexec --layerDeviceTypes / --layerPrecisions 를 사용해
    detection head 레이어를 GPU FP16, 나머지를 DLA INT8 로 빌드.
    """
    # detection head 레이어: GPU FP16 강제
    device_flag    = ",".join(f"{n}:GPU"  for n in head_layers)
    precision_flag = ",".join(f"{n}:fp16" for n in head_layers)

    cmd = [
        TRTEXEC,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--int8", "--fp16",
        f"--useDLACore={dla_core}",
        "--allowGPUFallback",
        f"--calib={calib_cache}",
        # detection head 레이어를 GPU FP16 으로 고정
        "--precisionConstraints=obey",
        f"--layerDeviceTypes={device_flag}",
        f"--layerPrecisions={precision_flag}",
        # 빌드 최적화
        f"--builderOptimizationLevel={BUILDER_OPT}",
        f"--timingCacheFile={timing_cache}",
        f"--memPoolSize=workspace:{WORKSPACE_GB * 1024}MiB",
        f"--avgTiming={AVG_TIMING}",
    ]

    print(f"\n[DLA Core {dla_core}] 혼합 정밀도 엔진 빌드 중...")
    print(f"  ONNX   : {onnx_path}")
    print(f"  엔진   : {engine_path}")
    print(f"  전략   : backbone/neck → DLA INT8,  detection head {len(head_layers)}개 레이어 → GPU FP16")
    print()

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"trtexec 실패 (exit={ret.returncode})")

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"\n  완료: {engine_path}  ({size_mb:.1f} MB)")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.chdir(WORK_DIR)

    model     = args.model
    sz        = args.input_size
    dla_core  = args.dla_core
    pattern   = args.head_pattern
    onnx_path = f"{model}_{sz}_split.onnx"

    print("=" * 60)
    print("YOLOv8m Mixed Precision Engine Builder")
    print(f"  Backbone/Neck : DLA Core N  INT8")
    print(f"  Detection Head: GPU         FP16")
    print("=" * 60)
    print(f"\nONNX : {onnx_path}")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX 파일 없음: {onnx_path}\n"
            "  → python3 export_yolov8.py 를 먼저 실행하세요."
        )

    # Detection head 레이어 식별
    print("\n[1/2] Detection head 레이어 식별 중...")
    head_layers = identify_head_layers(onnx_path, pattern)

    # --inspect 모드: 레이어 목록만 출력
    if args.inspect:
        print(f"\nDetection head 레이어 ({len(head_layers)}개):")
        for i, name in enumerate(head_layers, 1):
            print(f"  {i:3d}. {name}")
        return

    if not head_layers:
        print("\n  ⚠  Detection head 레이어를 찾지 못했습니다.")
        print("     --head-pattern 옵션으로 올바른 패턴을 지정하거나")
        print("     --inspect 로 전체 레이어명을 확인하세요.")
        sys.exit(1)

    # dla-core -1: 검사만
    if dla_core < 0:
        print("\n  --dla-core 를 0 또는 1 로 지정해야 엔진을 빌드합니다.")
        print(f"  검출된 detection head 레이어 수: {len(head_layers)}개")
        return

    # 캘리브레이션 캐시 확인
    calib_cache = f"{model}_{sz}_int8_calib.cache"
    if not os.path.exists(calib_cache):
        raise FileNotFoundError(
            f"캘리브레이션 캐시 없음: {calib_cache}\n"
            "  → python3 build_int8_engine.py --dla-core -1 을 먼저 실행하세요."
        )

    timing_cache = f"{model}_{sz}_dla{dla_core}_mixed_timing.cache"
    engine_path  = f"{model}_{sz}_b{BATCH_SIZE}_dla{dla_core}_mixed.engine"

    # 엔진 빌드
    print(f"\n[2/2] 엔진 빌드...")
    build_engine_mixed(
        onnx_path, engine_path, dla_core,
        calib_cache, timing_cache, head_layers,
    )

    print("\n다음 단계:")
    print(f"  DeepStream 실행: python3 deepstream_yolov8_4ch_dla.py")
    print(f"  (config_infer_yolov8_dla{dla_core}_mixed.txt 사용)")


if __name__ == "__main__":
    main()
