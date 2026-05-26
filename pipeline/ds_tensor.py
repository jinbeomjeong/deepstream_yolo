import ctypes

import numpy as np
import pyds


def read_layer_arr(layer, n_elem):
    """layer.buffer 에서 n_elem 개 요소를 float32 ndarray 로 반환."""
    if layer.dataType == pyds.HALF:
        ptr = ctypes.cast(pyds.get_ptr(layer.buffer),
                          ctypes.POINTER(ctypes.c_uint16))
        return (np.ctypeslib.as_array(ptr, shape=(n_elem,))
                .view(np.float16).astype(np.float32))
    else:
        ptr = ctypes.cast(pyds.get_ptr(layer.buffer),
                          ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(ptr, shape=(n_elem,)).copy()


def parse_layer(layer, n_frames):
    """
    NvDsInferLayerInfo 에서 배치 텐서를 읽어 [n_frames, 300, 6] 로 반환.
    inferDims = [300, 6]          → 버퍼에 n_frames × n_elem 연속 저장
    inferDims = [n_frames, 300, 6] → 버퍼에 n_elem 저장 (배치 포함)
    """
    dims = layer.inferDims
    shape = [dims.d[j] for j in range(dims.numDims)]
    n_elem = 1
    for s in shape:
        n_elem *= s

    if len(shape) == 3 and shape[0] == n_frames:
        # inferDims 자체에 배치 차원 포함: [n_frames, 300, 6]
        arr = read_layer_arr(layer, n_elem)
        return arr.reshape(shape), shape        # [n_frames, 300, 6]
    else:
        # inferDims = per-frame: [300, 6]
        # 버퍼에 n_frames × n_elem 연속 저장
        arr = read_layer_arr(layer, n_frames * n_elem)
        return arr.reshape([n_frames] + shape), shape   # [n_frames, 300, 6]


def get_tensor_from_user_meta(user_meta_list):
    """GList(user_meta)를 순회해 첫 번째 TENSOR_OUTPUT_META 의 layer 반환."""
    l = user_meta_list
    while l is not None:
        try:
            um = pyds.NvDsUserMeta.cast(l.data)
        except StopIteration:
            break
        if um.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            return pyds.get_nvds_LayerInfo(tm, 0)
        try:
            l = l.next
        except StopIteration:
            break
    return None


def parse_yolo11(tensor, src_w, src_h, conf_thresh, infer_w=640, infer_h=640):
    """
    tensor shape: [300, 6] (GPU에서 NMS 처리가 완료된 결과)
    col 0-3: x1, y1, x2, y2 (Letterbox 적용된 infer_w×infer_h 기준 픽셀)
    col 4  : confidence
    col 5  : class_id
    """
    scale = min(infer_w / src_w, infer_h / src_h)
    pad_x = (infer_w - src_w * scale) / 2
    pad_y = (infer_h - src_h * scale) / 2

    dets = []
    for i in range(tensor.shape[0]):
        box = tensor[i]
        conf = box[4]
        if conf < conf_thresh or conf == 0:
            continue
        class_id = int(box[5])
        x1 = np.clip((box[0] - pad_x) / scale, 0, src_w)
        y1 = np.clip((box[1] - pad_y) / scale, 0, src_h)
        x2 = np.clip((box[2] - pad_x) / scale, 0, src_w)
        y2 = np.clip((box[3] - pad_y) / scale, 0, src_h)
        dets.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "conf": float(conf), "class_id": class_id,
        })
    return dets
