#!/usr/bin/env python3

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from ultralytics import YOLO

MODEL_PATH = "yolo11m.pt"
NMS_OP_TYPES = {
    "NonMaxSuppression",
    "BatchedNMS_TRT",
    "EfficientNMS_TRT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO11m to ONNX without NMS nodes."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="ONNX export batch size. Default: 4",
    )
    return parser.parse_args()


def _tensor_shape(value_info) -> list[int | str]:
    return [
        dim.dim_value if dim.dim_value else dim.dim_param
        for dim in value_info.type.tensor_type.shape.dim
    ]


def split_final_concat_outputs(onnx_path: Path) -> None:
    model = onnx.load(onnx_path)
    graph = model.graph
    if len(graph.output) != 1:
        raise RuntimeError(
            f"Expected one ONNX output before split, got {len(graph.output)}"
        )

    output_name = graph.output[0].name
    final_concat = next(
        (
            node
            for node in graph.node
            if output_name in node.output and node.op_type == "Concat"
        ),
        None,
    )
    if final_concat is None:
        raise RuntimeError(f"Final Concat node for output '{output_name}' not found")
    if len(final_concat.input) != 2:
        raise RuntimeError(
            f"Expected final Concat to have 2 inputs, got {len(final_concat.input)}"
        )

    axis = next((attr.i for attr in final_concat.attribute if attr.name == "axis"), 1)
    if axis not in (1, -2):
        raise RuntimeError(f"Unexpected final Concat axis: {axis}")

    output_shape = _tensor_shape(graph.output[0])
    if len(output_shape) != 3:
        raise RuntimeError(f"Expected [B, 84, A] output shape, got {output_shape}")

    batch, channels, anchors = output_shape
    if channels != 84:
        raise RuntimeError(f"Expected 84 output channels, got {channels}")

    boxes_name, classes_name = final_concat.input
    graph.node.remove(final_concat)
    graph.output.clear()
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                boxes_name, TensorProto.FLOAT, [batch, 4, anchors]
            ),
            helper.make_tensor_value_info(
                classes_name, TensorProto.FLOAT, [batch, channels - 4, anchors]
            ),
        ]
    )

    graph.output[0].name = "output_boxes"
    graph.output[1].name = "output_classes"
    for node in graph.node:
        for i, output in enumerate(node.output):
            if output == boxes_name:
                node.output[i] = "output_boxes"
            elif output == classes_name:
                node.output[i] = "output_classes"

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def assert_nms_removed(onnx_path: Path) -> None:
    model = onnx.load(onnx_path)
    nms_nodes = [
        node
        for node in model.graph.node
        if node.op_type in NMS_OP_TYPES or "NMS" in node.op_type.upper()
    ]
    if nms_nodes:
        details = ", ".join(
            f"{node.name or '<unnamed>'}:{node.op_type}" for node in nms_nodes
        )
        raise RuntimeError(f"ONNX still contains NMS nodes: {details}")

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


args = parse_args()
onnx_path = Path(f"yolo11m_b{args.batch}.onnx")
default_export_path = Path(MODEL_PATH).with_suffix(".onnx")
if default_export_path.exists() and default_export_path.resolve() != onnx_path.resolve():
    default_export_path.unlink()

model = YOLO(MODEL_PATH)  # 없으면 자동 다운로드

exported_path = Path(model.export(
    format="onnx",
    imgsz=640,
    batch=args.batch,
    opset=17,
    simplify=False,
    dynamic=False,
    nms=False,
    half=False,
))

if exported_path.resolve() != onnx_path.resolve():
    exported_path.replace(onnx_path)

split_final_concat_outputs(onnx_path)
assert_nms_removed(onnx_path)

print(f"export complete {onnx_path} (batch={args.batch})")
