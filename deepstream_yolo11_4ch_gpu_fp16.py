#!/usr/bin/env python3
"""
DeepStream 4채널 YOLO11m 추론 파이프라인 (GPU CUDA, FP16)

실행:
  source /home/nvidia/workspace/arround_view/venv/bin/activate
  python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... [video4]                  # 헤드리스
  DISPLAY=:0 python deepstream_yolo11_4ch_gpu_fp16.py [video1] ...                # 화면 출력
  python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... --output out.mp4          # 영상 저장
  DISPLAY=:0 python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... -o out.mp4     # 화면 + 저장

  - 인자 없음: 기본 샘플 영상 4채널
  - 1~4개 인자: 채널 순서대로 각 영상 할당
  - --output FILE (-o FILE): 후처리 완료 영상을 HW(nvv4l2h264enc) 인코딩으로 저장

파이프라인:
  4x source → nvstreammux → nvinfer(GPU FP16) → [probe: YOLO 파싱] → tiler → OSD → sink
                                                                                  ↘ (--output) HW encoder → mp4
"""

import sys
import os
import argparse
import ctypes
import time
import signal
import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import pyds

# ── 설정 ────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="DeepStream 4채널 YOLO11m FP16 파이프라인")
_parser.add_argument("sources", nargs="*", metavar="VIDEO",
                     help="입력 영상 경로 (최대 4개, 생략 시 기본 샘플)")
_parser.add_argument("--output", "-o", metavar="FILE", default=None,
                     help="HW 인코딩으로 저장할 출력 영상 경로 (.mp4)")
_parsed = _parser.parse_args()

_DEFAULT_SRC  = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
VIDEO_SOURCES = _parsed.sources[:4] if _parsed.sources else [_DEFAULT_SRC] * 4
NUM_SOURCES   = len(VIDEO_SOURCES)
OUTPUT_PATH   = _parsed.output
SAVE_VIDEO    = OUTPUT_PATH is not None
PGIE_CONFIG   = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolo11_gpu_fp16.txt"
MUXER_W       = 1920
MUXER_H       = 1080
CONF_THRESH   = 0.30
IOU_THRESH    = 0.45
INFER_W, INFER_H = 640, 640
USE_DISPLAY   = bool(os.environ.get("DISPLAY"))

COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# ── 전역 상태 ────────────────────────────────────────────────────────────────────
_frame_count  = 0
_t_start      = time.time()
_win_t        = time.time()   # 슬라이딩 윈도우 시작 시각
_win_frames   = 0             # 윈도우 내 프레임 수
_diag_printed = False   # 텐서 구조 진단 (최초 1회)


# ── YOLO11 텐서 파싱 (NMS 내장 엔진용) ──────────────────────────────────────────
def _parse_yolo11(tensor: np.ndarray, src_w: int, src_h: int):
    """
    tensor shape: [300, 6] (GPU에서 NMS 처리가 완료된 결과)
    col 0-3: x1, y1, x2, y2 (Letterbox 적용된 640x640 기준 픽셀)
    col 4  : confidence (확률)
    col 5  : class_id (사물 종류)
    """
    scale = min(INFER_W / src_w, INFER_H / src_h)
    pad_x = (INFER_W - src_w * scale) / 2
    pad_y = (INFER_H - src_h * scale) / 2

    dets = []

    # 300개의 결과물을 하나씩 꺼내어 확인
    for i in range(tensor.shape[0]):
        box = tensor[i]
        conf = box[4]

        # 확률이 임계값보다 낮거나 0인 빈 데이터는 버림
        if conf < CONF_THRESH or conf == 0:
            continue

        class_id = int(box[5])

        # GPU가 넘겨준 좌표 (x1, y1, x2, y2)
        x1_raw, y1_raw, x2_raw, y2_raw = box[0], box[1], box[2], box[3]

        # 화면 여백(블랙바)을 빼고, 원래 영상 비율로 좌표 복원
        x1 = np.clip((x1_raw - pad_x) / scale, 0, src_w)
        y1 = np.clip((y1_raw - pad_y) / scale, 0, src_h)
        x2 = np.clip((x2_raw - pad_x) / scale, 0, src_w)
        y2 = np.clip((y2_raw - pad_y) / scale, 0, src_h)

        dets.append({
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "conf": float(conf), "class_id": class_id
        })

    return dets


# ── NvDsObjectMeta 추가 ───────────────────────────────────────────────────────
def _add_obj_meta(batch_meta, frame_meta, det):
    obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj.class_id   = det["class_id"]
    obj.confidence = det["conf"]
    obj.object_id  = 0xFFFFFFFFFFFFFFFF   # UNTRACKED

    c = obj.detector_bbox_info.org_bbox_coords
    c.left, c.top = det["x1"], det["y1"]
    c.width  = det["x2"] - det["x1"]
    c.height = det["y2"] - det["y1"]

    r = obj.rect_params
    r.left, r.top = det["x1"], det["y1"]
    r.width, r.height = c.width, c.height
    r.border_width = 2
    r.border_color.set(0.0, 1.0, 0.0, 1.0)

    label = COCO_LABELS[det["class_id"]] if det["class_id"] < len(COCO_LABELS) else str(det["class_id"])
    t = obj.text_params
    t.display_text = f"{label} {det['conf']:.2f}"
    t.x_offset, t.y_offset = int(det["x1"]), max(0, int(det["y1"]) - 20)
    t.font_params.font_name = "Serif"
    t.font_params.font_size = 12
    t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    t.set_bg_clr = 1
    t.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)

# ── 레이어 버퍼 읽기 ──────────────────────────────────────────────────────────
def _read_layer_arr(layer, n_elem):
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

# ── 텐서 레이어 파싱 헬퍼 ────────────────────────────────────────────────────
def _parse_layer(layer, n_frames):
    """
    NvDsInferLayerInfo 에서 배치 텐서를 읽어 [n_frames, 300, 6] 로 반환.
    inferDims = [300, 6]         → 버퍼에 n_frames × n_elem 연속 저장
    inferDims = [n_frames,300,6] → 버퍼에 n_elem 저장 (배치 포함)
    """
    dims  = layer.inferDims
    shape = [dims.d[j] for j in range(dims.numDims)]
    n_elem = 1
    for s in shape:
        n_elem *= s

    if len(shape) == 3 and shape[0] == n_frames:
        # inferDims 자체에 배치 차원 포함: [n_frames, 84, 8400]
        arr = _read_layer_arr(layer, n_elem)
        return arr.reshape(shape), shape        # [n_frames, 84, 8400]
    else:
        # inferDims = per-frame: [84, 8400]
        # 버퍼에 n_frames × n_elem 연속 저장
        arr = _read_layer_arr(layer, n_frames * n_elem)
        return arr.reshape([n_frames] + shape), shape   # [n_frames, 84, 8400]


def _get_tensor_from_user_meta(user_meta_list):
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


# ── pgie src pad 프로브 ───────────────────────────────────────────────────────
def pgie_src_pad_probe(pad, info, u_data):
    """
    탐색 우선순위:
      1) batch_meta.batch_user_meta_list   ← primary GIE 배치 모드
      2) 각 frame_meta.frame_user_meta_list (Case A: 프레임별 독립 텐서)
      3) frame[0] 의 frame_user_meta_list   (Case B: 배치 전체가 frame 0 버퍼에)

    최초 실행 시 어느 경로에서 텐서를 얻었는지 진단 메시지를 출력한다.
    """
    global _frame_count, _det_count, _diag_printed
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    # ── 1) 모든 frame_meta 수집 (리스트 인덱스 = 배치 순서) ─────────────────
    frames = []
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frames.append(pyds.NvDsFrameMeta.cast(l_frame.data))
        except StopIteration:
            break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    n_frames = len(frames)
    if n_frames == 0:
        return Gst.PadProbeReturn.OK

    tensors = [None] * n_frames   # tensors[i]: np.ndarray [300, 6]
    diag_source = "none"

    # ── 2) batch_user_meta_list 우선 탐색 ────────────────────────────────────
    layer = _get_tensor_from_user_meta(batch_meta.batch_user_meta_list)
    if layer is not None:
        batch, shape = _parse_layer(layer, n_frames)
        for i in range(n_frames):
            tensors[i] = batch[i]
        diag_source = f"batch_user_meta | inferDims={shape}"

    # ── 3) frame_user_meta_list 탐색 (배치 레벨에 없을 때) ──────────────────
    if tensors[0] is None:
        frame_layers = [_get_tensor_from_user_meta(fm.frame_user_meta_list)
                        for fm in frames]
        n_found = sum(1 for x in frame_layers if x is not None)

        if n_found == n_frames:
            # 케이스 A: 모든 프레임에 독립 텐서
            for i, lyr in enumerate(frame_layers):
                if lyr is None:
                    continue
                dims  = lyr.inferDims
                shape = [dims.d[j] for j in range(dims.numDims)]
                n_elem = 1
                for s in shape:
                    n_elem *= s
                tensors[i] = _read_layer_arr(lyr, n_elem).reshape(shape)
            diag_source = f"frame_user_meta Case-A | inferDims={shape}"

        elif n_found > 0:
            # 케이스 B: frame 0 버퍼에 배치 전체 저장
            first = next(i for i, x in enumerate(frame_layers) if x is not None)
            batch, shape = _parse_layer(frame_layers[first], n_frames)
            for i in range(n_frames):
                tensors[i] = batch[i]
            diag_source = (f"frame_user_meta Case-B "
                           f"({n_found}/{n_frames} frames have meta) | inferDims={shape}")

    # ── 4) 최초 1회 진단 출력 ────────────────────────────────────────────────
    if not _diag_printed:
        _diag_printed = True
        print(f"[진단] n_frames={n_frames} | 텐서 출처={diag_source}")
        for i, fm in enumerate(frames):
            valid = tensors[i] is not None
            conf  = float(tensors[i][:, 4].max()) if valid else 0.0
            print(f"  [idx={i}] src={fm.source_id} batch_id={fm.batch_id} "
                  f"tensor={'유효' if valid else '없음'} max_conf={conf:.3f}")

    if all(t is None for t in tensors):
        return Gst.PadProbeReturn.OK

    # ── 5) 각 프레임에 detections 적용 ──────────────────────────────────────
    global _win_t, _win_frames

    for i, fm in enumerate(frames):
        if tensors[i] is None:
            continue

        dets = _parse_yolo11(tensors[i], MUXER_W, MUXER_H)

        _frame_count += 1
        _win_frames  += 1

        for det in dets:
            _add_obj_meta(batch_meta, fm, det)

        if _frame_count % 100 == 0:
            now     = time.time()
            elapsed = now - _win_t
            fps     = _win_frames / elapsed if elapsed > 0 else 0
            print(f"[{_frame_count:6d}프레임] FPS={fps:.1f}")
            _win_t      = now
            _win_frames = 0

    return Gst.PadProbeReturn.OK

# ── 버스 콜백 ─────────────────────────────────────────────────────────────────
pipeline = None

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("EOS — 영상 종료, 파이프라인을 정리합니다.")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"ERROR: {err}: {debug}\n")
        loop.quit()
    return True

# ── 소스 생성 ─────────────────────────────────────────────────────────────────
def make_src_and_connect(idx, path, mux, pipeline):
    udbin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-{idx}")
    udbin.set_property("uri", f"file://{os.path.abspath(path)}")
    pipeline.add(udbin)

    sink_pad = mux.request_pad_simple(f"sink_{idx}")

    def cb_newpad(dec, pad, sink_pad):
        caps     = pad.get_current_caps()
        gstname  = caps.get_structure(0).get_name() if caps else ""
        features = caps.get_features(0) if caps else None
        if "video" in gstname and features and features.contains("memory:NVMM"):
            if not sink_pad.is_linked():
                if pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                    sys.stderr.write(f"소스 {idx}: pad 링크 실패\n")

    udbin.connect("pad-added", cb_newpad, sink_pad)

# ── HW 인코더 저장 브랜치 ────────────────────────────────────────────────────
def _link_save_branch(pipeline, src_el):
    """src_el 뒤에 nvv4l2h264enc HW 인코더 → mp4 저장 브랜치를 연결한다."""
    conv_enc  = Gst.ElementFactory.make("nvvideoconvert", "conv-enc")
    encoder   = Gst.ElementFactory.make("nvv4l2h264enc",  "encoder")
    h264parse = Gst.ElementFactory.make("h264parse",      "h264-parse")
    mux_mp4   = Gst.ElementFactory.make("mp4mux",         "mp4-mux")
    filesink  = Gst.ElementFactory.make("filesink",       "filesink")

    encoder.set_property("bitrate",       8000000)   # 8 Mbps
    encoder.set_property("preset-level", 1)          # UltraFast (HW 최고속)
    encoder.set_property("idrinterval",  60)          # IDR 60프레임(2초) — 표준 GOP, HW 디코더 리셋 최소화
    filesink.set_property("location",    OUTPUT_PATH)
    filesink.set_property("sync",        False)

    for el in (conv_enc, encoder, h264parse, mux_mp4, filesink):
        pipeline.add(el)

    src_el.link(conv_enc)
    conv_enc.link(encoder)
    encoder.link(h264parse)
    h264parse.link(mux_mp4)
    mux_mp4.link(filesink)


# ── 파이프라인 구성 ───────────────────────────────────────────────────────────
def main():
    global pipeline
    Gst.init(None)
    pipeline = Gst.Pipeline()

    # nvstreammux
    mux = Gst.ElementFactory.make("nvstreammux", "muxer")
    mux.set_property("width",                MUXER_W)
    mux.set_property("height",               MUXER_H)
    mux.set_property("batch-size",           NUM_SOURCES)
    mux.set_property("batched-push-timeout", 40000)
    pipeline.add(mux)

    for i, src in enumerate(VIDEO_SOURCES):
        make_src_and_connect(i, src, mux, pipeline)

    # nvinfer (YOLO11m GPU FP16)
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pipeline.add(pgie)

    mux.link(pgie)

    if USE_DISPLAY or SAVE_VIDEO:
        tiler    = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
        conv_osd = Gst.ElementFactory.make("nvvideoconvert",     "conv-osd")
        osd      = Gst.ElementFactory.make("nvdsosd",            "osd")
        conv_out = Gst.ElementFactory.make("nvvideoconvert",     "conv-out")

        tiler_cols = min(NUM_SOURCES, 2)
        tiler_rows = (NUM_SOURCES + tiler_cols - 1) // tiler_cols
        tiler.set_property("rows",    tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width",   1920)
        tiler.set_property("height",  1080)
        osd.set_property("process-mode", 1)

        for el in (tiler, conv_osd, osd, conv_out):
            pipeline.add(el)

        pgie.link(tiler)
        tiler.link(conv_osd)
        conv_osd.link(osd)
        osd.link(conv_out)

        if USE_DISPLAY and SAVE_VIDEO:
            # 화면 출력 + 저장: tee로 분기
            tee    = Gst.ElementFactory.make("tee",     "tee")
            q_disp = Gst.ElementFactory.make("queue",   "q-disp")
            q_save = Gst.ElementFactory.make("queue",   "q-save")
            sink   = Gst.ElementFactory.make("nv3dsink","sink")
            sink.set_property("sync", False)

            for el in (tee, q_disp, q_save, sink):
                pipeline.add(el)

            conv_out.link(tee)
            tee.request_pad_simple("src_%u").link(q_disp.get_static_pad("sink"))
            q_disp.link(sink)
            tee.request_pad_simple("src_%u").link(q_save.get_static_pad("sink"))
            _link_save_branch(pipeline, q_save)
            print(f"출력: nv3dsink {tiler_rows}×{tiler_cols} 타일 디스플레이 + 저장 → {OUTPUT_PATH}")

        elif USE_DISPLAY:
            sink = Gst.ElementFactory.make("nv3dsink", "sink")
            sink.set_property("sync", False)
            pipeline.add(sink)
            conv_out.link(sink)
            print(f"출력: nv3dsink {tiler_rows}×{tiler_cols} 타일 디스플레이")

        else:
            # 저장 전용: 렌더링은 하되 화면 출력 없음
            _link_save_branch(pipeline, conv_out)
            print(f"출력: 저장 전용 → {OUTPUT_PATH}")

    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        pgie.link(sink)
        print("출력: fakesink (헤드리스) — 화면 출력은 DISPLAY=:0, 저장은 --output 사용")

    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, pgie_src_pad_probe, 0
    )

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # SIGTERM/SIGINT → EOS 전송 → mp4mux moov 정상 기록 후 종료
    def _graceful_stop(signum, frame):
        print("\n종료 신호 수신 — EOS 전송 중 (파일 마무리)...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT,  _graceful_stop)

    for i, src in enumerate(VIDEO_SOURCES):
        print(f"소스[{i}]: {src}")
    print(f"YOLO11m GPU FP16 파이프라인 시작 ({NUM_SOURCES}채널, Ctrl+C 종료)...")
    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    elapsed = time.time() - _t_start
    fps = _frame_count / elapsed if elapsed > 0 else 0
    print(f"\n총 처리: {_frame_count}프레임 | 평균 FPS={fps:.1f}")
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
