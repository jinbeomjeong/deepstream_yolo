#!/usr/bin/env python3
"""
YOLO ONNX 최종 Concat 제거: output0 [B,84,N] → output_boxes [B,4,N] + output_classes [B,80,N]

방식: 최종 Concat 노드를 제거하고 그 입력(box decode, class sigmoid)을 직접 출력으로 등록.
Concat 이전 단계에서 분리하므로 INT8 스케일이 각 텐서에 독립 적용됨.

사용:
  python3 split_yolo_output.py yolov8m_512.onnx
  python3 split_yolo_output.py yolov8m_640_b200_calib.onnx   →  yolov8m_640_b200_calib_split.onnx
"""

import argparse
import onnx
from onnx import helper, TensorProto


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO ONNX 최종 Concat 제거: output0 → output_boxes + output_classes"
    )
    parser.add_argument("src", metavar="INPUT_ONNX",
                        help="입력 ONNX 파일")
    return parser.parse_args()


args = parse_args()
src = args.src
dst = src.replace(".onnx", "_split.onnx")

model = onnx.load(src)
graph = model.graph

# output0 을 생성하는 최종 Concat 노드 찾기
orig_out_name = graph.output[0].name   # "output0"
concat_node = None
for node in graph.node:
    if node.op_type == "Concat" and orig_out_name in node.output:
        concat_node = node
        break

if concat_node is None:
    raise RuntimeError(f"output0 을 생성하는 Concat 노드를 찾을 수 없음")

print(f"Concat 노드: {concat_node.name}")
print(f"  입력: {list(concat_node.input)}")

# Concat 입력 두 개: boxes 텐서와 classes 텐서 식별
# 각 입력의 shape 을 value_info 에서 조회
shape_map = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    shape_map[vi.name] = dims

boxes_name   = None
classes_name = None
for inp in concat_node.input:
    dims = shape_map.get(inp, [])
    ch = dims[1] if len(dims) >= 2 else -1
    if ch == 4:
        boxes_name   = inp
    elif ch > 4:
        classes_name = inp

if boxes_name is None or classes_name is None:
    # shape 정보 없을 경우 순서로 추정 (boxes 먼저)
    boxes_name, classes_name = concat_node.input[0], concat_node.input[1]
    print("  shape 정보 없음 → 순서로 추정 (boxes=input[0], classes=input[1])")

print(f"  boxes   → {boxes_name}")
print(f"  classes → {classes_name}")

# Concat 노드 제거
graph.node.remove(concat_node)

# output0 제거, output_boxes / output_classes 등록
del graph.output[:]

# shape 추론
orig_shape = [d.dim_value for d in onnx.load(src).graph.output[0].type.tensor_type.shape.dim]
B, _, N = orig_shape

graph.output.extend([
    helper.make_tensor_value_info("output_boxes",   TensorProto.FLOAT, [B, 4,  N]),
    helper.make_tensor_value_info("output_classes", TensorProto.FLOAT, [B, 80, N]),
])

# 그래프 내부 텐서명을 output_boxes / output_classes 로 맞추는 Identity 노드 추가
graph.node.append(helper.make_node("Identity", [boxes_name],   ["output_boxes"],   "id_boxes"))
graph.node.append(helper.make_node("Identity", [classes_name], ["output_classes"], "id_classes"))

onnx.checker.check_model(model)
onnx.save(model, dst)
print(f"\n저장: {dst}")
print(f"  output_boxes   : [{B}, 4,  {N}]")
print(f"  output_classes : [{B}, 80, {N}]")
