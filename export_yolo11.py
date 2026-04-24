#!/usr/bin/env python3

from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # 없으면 자동 다운로드

model.export(
    format="onnx",
    imgsz=640,
    batch=4,          # nvstreammux batch-size 와 일치
    opset=17,
    simplify=True,
    dynamic=False,
    nms=True,
    half=False,
)

print("완료: yolo11m.onnx")
