#!/usr/bin/env python3
"""
YOLO 모델 → ONNX 내보내기

build_int8_engine.py 에서 필요한 ONNX 파일 두 가지를 한 번에 생성:
  - {MODEL}.onnx              : 엔진용 (batch=BATCH_SIZE)
  - {MODEL}_b{N}_calib.onnx  : 캘리브레이션용 (batch=CALIB_BATCH_SIZE)

[실행]
  MODEL=yolov8m  python3 export_yolo11.py
  MODEL=yolo11m  python3 export_yolo11.py   (기본값)
"""

import os
from ultralytics import YOLO

# build_int8_engine.py 와 동일한 상수
MODEL            = os.environ.get("MODEL",      "yolo11m")
INPUT_SIZE       = int(os.environ.get("INPUT_SIZE", "640"))
BATCH_SIZE       = 2     # 엔진 배치 크기
CALIB_BATCH_SIZE = 100   # 캘리브레이션 배치 크기

ENGINE_ONNX = f"{MODEL}_{INPUT_SIZE}.onnx"
CALIB_ONNX  = f"{MODEL}_{INPUT_SIZE}_b{CALIB_BATCH_SIZE}_calib.onnx"

EXPORT_KWARGS = dict(
    format="onnx",
    imgsz=INPUT_SIZE,
    opset=17,
    simplify=True,
    dynamic=False,
    nms=False,
    half=False,
)

print(f"모델: {MODEL}.pt  imgsz={INPUT_SIZE}  →  {ENGINE_ONNX}, {CALIB_ONNX}")
model = YOLO(f"{MODEL}.pt")

# ── 1. 캘리브레이션용 ONNX (batch=CALIB_BATCH_SIZE) ──────────────────────────
# 캘리브레이션을 먼저 export 후 rename → 이후 엔진 ONNX 가 덮어쓰지 않도록
print(f"\n[1/2] 캘리브레이션 ONNX: {CALIB_ONNX}  (batch={CALIB_BATCH_SIZE})")
model.export(batch=CALIB_BATCH_SIZE, **EXPORT_KWARGS)
os.rename(f"{MODEL}.onnx", CALIB_ONNX)
print(f"      완료: {CALIB_ONNX}")

# ── 2. 엔진용 ONNX (batch=BATCH_SIZE) ────────────────────────────────────────
print(f"\n[2/2] 엔진 ONNX: {ENGINE_ONNX}  (batch={BATCH_SIZE})")
model.export(batch=BATCH_SIZE, **EXPORT_KWARGS)
os.rename(f"{MODEL}.onnx", ENGINE_ONNX)
print(f"      완료: {ENGINE_ONNX}")

print(f"\n내보내기 완료:")
print(f"  {ENGINE_ONNX:<35} → 엔진 빌드용")
print(f"  {CALIB_ONNX:<35} → 캘리브레이션용")
