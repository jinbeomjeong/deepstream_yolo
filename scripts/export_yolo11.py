#!/usr/bin/env python3
import os
import shutil

from ultralytics import YOLO

_SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR   = os.path.join(os.path.dirname(_SCRIPTS_DIR), "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

model = YOLO("yolo11m.pt")  # 없으면 자동 다운로드

result = model.export(
    format="onnx",
    imgsz=640,
    batch=4,          # nvstreammux batch-size 와 일치
    opset=17,
    simplify=True,
    dynamic=False,
    nms=True,
    half=False,
)

src = str(result)
dst = os.path.join(_MODELS_DIR, os.path.basename(src))
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.move(src, dst)

print(f"완료: {dst}")
