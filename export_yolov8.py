#!/usr/bin/env python3
"""
YOLO 모델 → ONNX 내보내기

build_int8_engine.py 에서 필요한 ONNX 파일 두 가지를 한 번에 생성:
  - {MODEL}_{SIZE}.onnx              : 엔진용 (batch=2)
  - {MODEL}_{SIZE}_b100_calib.onnx  : 캘리브레이션용 (batch=100)

[실행]
  python3 export_yolov8.py                                        (기본값: yolov8m, 640, calib-batch=100)
  python3 export_yolov8.py --model yolo11m
  python3 export_yolov8.py --model yolov8m --input-size 512
  python3 export_yolov8.py --calib-batch-size 500
"""

import os
import argparse
from ultralytics import YOLO

BATCH_SIZE = 2     # 엔진 배치 크기


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 모델 → ONNX 내보내기")
    parser.add_argument("--model", default="yolov8m",
                        help="모델 이름 (기본값: yolov8m)")
    parser.add_argument("--input-size", type=int, default=640, metavar="N",
                        help="입력 해상도 (기본값: 640)")
    parser.add_argument("--calib-batch-size", type=int, default=100, metavar="N",
                        help="캘리브레이션 ONNX 배치 크기 (기본값: 100)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name       = args.model
    input_size       = args.input_size
    calib_batch_size = args.calib_batch_size

    engine_onnx = f"{model_name}_{input_size}.onnx"
    calib_onnx  = f"{model_name}_{input_size}_b{calib_batch_size}_calib.onnx"

    export_kwargs = dict(
        format="onnx",
        imgsz=input_size,
        opset=17,
        simplify=True,
        dynamic=False,
        nms=False,
        half=False,
    )

    print(f"모델: {model_name}.pt  imgsz={input_size}  →  {engine_onnx}, {calib_onnx}")
    model = YOLO(f"{model_name}.pt")

    # ── 1. 캘리브레이션용 ONNX (batch=CALIB_BATCH_SIZE) ──────────────────────────
    # 캘리브레이션을 먼저 export 후 rename → 이후 엔진 ONNX 가 덮어쓰지 않도록
    print(f"\n[1/2] 캘리브레이션 ONNX: {calib_onnx}  (batch={calib_batch_size})")
    model.export(batch=calib_batch_size, **export_kwargs)
    os.rename(f"{model_name}.onnx", calib_onnx)
    print(f"      완료: {calib_onnx}")

    # ── 2. 엔진용 ONNX (batch=BATCH_SIZE) ────────────────────────────────────────
    print(f"\n[2/2] 엔진 ONNX: {engine_onnx}  (batch={BATCH_SIZE})")
    model.export(batch=BATCH_SIZE, **export_kwargs)
    os.rename(f"{model_name}.onnx", engine_onnx)
    print(f"      완료: {engine_onnx}")

    print(f"\n내보내기 완료:")
    print(f"  {engine_onnx:<35} → 엔진 빌드용")
    print(f"  {calib_onnx:<35} → 캘리브레이션용")


if __name__ == "__main__":
    main()
