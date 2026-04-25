#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # 없으면 자동 다운로드

model.export(
    format="onnx",
    imgsz=640,
    batch=4,
    opset=17,
    simplify=True,
    dynamic=False,
    nms=False,
    half=False,
)

print("export complete yolo11m.onnx")
