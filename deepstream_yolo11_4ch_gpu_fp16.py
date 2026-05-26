#!/usr/bin/env python3
"""
DeepStream 4채널 YOLO11m 추론 파이프라인 (GPU CUDA, FP16)

실행:
  source /home/nvidia/workspace/arround_view/venv/bin/activate
  python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... [video4]                  # 헤드리스
  DISPLAY=:0 python deepstream_yolo11_4ch_gpu_fp16.py [video1] ...                # 화면 출력
  python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... --output out.mp4          # 영상 저장
  DISPLAY=:0 python deepstream_yolo11_4ch_gpu_fp16.py [video1] ... -o out.mp4     # 화면 + 저장

파이프라인:
  4x source → nvstreammux → nvinfer(GPU FP16, C++ 파서) → tiler → nvdsosd → sink
                                                                             ↘ (--output) HW encoder → mp4

탐지 흐름:
  nvinfer → lib_parser_yolo11.so (NvDsInferParseYolo11)
          → NvDsObjectMeta 자동 생성 (bbox + label)
          → nvdsosd 자동 렌더링

서브모듈 (pipeline/):
  parser_yolo11.cpp  — C++ 커스텀 파서 ([300,6] NMS 출력 처리)
  ds_pipeline.py     — GStreamer 엘리먼트 헬퍼 (소스, 저장 브랜치, 버스 콜백)
"""

import sys
import os
import argparse
import time
import signal

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from ds_pipeline import probe_video_size, bus_call, make_src_and_connect, link_save_branch

try:
    import pyds
    _PYDS_OK = True
except ImportError:
    _PYDS_OK = False
    print("[경고] pyds 모듈 없음 — obj_meta 진단 프로브 비활성화 (venv 활성화 필요)")

# ── 설정 ─────────────────────────────────────────────────────────────────────
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
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PGIE_CONFIG   = os.path.join(_PROJECT_ROOT, "config", "config_infer_yolo11_gpu_fp16.txt")
USE_DISPLAY   = bool(os.environ.get("DISPLAY"))

# ── FPS 카운터 ────────────────────────────────────────────────────────────────
# 탐지·bbox·라벨은 C++ 파서(lib_parser_yolo11.so)와 nvinfer/nvdsosd가 담당.
# Python 프로브는 처리 프레임 수와 FPS 계산만 수행한다.
_frame_count = 0
_t_start     = time.time()
_win_t       = time.time()
_win_frames  = 0

# ── obj_meta 진단 프로브 ──────────────────────────────────────────────────────
# nvinfer 출력단에서 NvDsObjectMeta 목록을 확인 — C++ 파서가 올바르게 동작하는지 검증.
# 처음 5회 배치(약 5프레임)만 출력 후 자동 비활성화.
_diag_batches = 0

def _obj_meta_probe(pad, info, u_data):
    global _diag_batches
    if not _PYDS_OK or _diag_batches >= 5:
        return Gst.PadProbeReturn.OK
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame:
        try:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        n_objs = 0
        l_obj = fm.obj_meta_list
        while l_obj:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                if n_objs == 0:
                    r = obj.rect_params
                    print(f"[진단] src={fm.source_id} frame={fm.frame_num} "
                          f"class={obj.class_id} conf={obj.confidence:.3f} "
                          f"left={r.left:.1f} top={r.top:.1f} "
                          f"w={r.width:.1f} h={r.height:.1f}")
                n_objs += 1
                l_obj = l_obj.next
            except StopIteration:
                break
        if n_objs == 0:
            print(f"[진단] src={fm.source_id} frame={fm.frame_num} — obj_meta 없음")
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    _diag_batches += 1
    return Gst.PadProbeReturn.OK

def _fps_probe(pad, info, u_data):
    global _frame_count, _win_t, _win_frames
    if not info.get_buffer():
        return Gst.PadProbeReturn.OK
    _frame_count += 1
    _win_frames  += 1
    if _frame_count % 100 == 0:
        now     = time.time()
        elapsed = now - _win_t
        fps     = _win_frames / elapsed if elapsed > 0 else 0
        print(f"[{_frame_count:6d}프레임] FPS={fps:.1f}")
        _win_t      = now
        _win_frames = 0
    return Gst.PadProbeReturn.OK

# ── 파이프라인 구성 ───────────────────────────────────────────────────────────
pipeline = None

def main():
    global pipeline
    Gst.init(None)

    # 첫 번째 소스의 해상도를 nvstreammux 출력 해상도로 사용
    muxer_w, muxer_h = probe_video_size(VIDEO_SOURCES[0])
    if not (muxer_w and muxer_h):
        muxer_w, muxer_h = 1920, 1080
        print(f"[경고] 해상도 탐색 실패 — 기본값 사용: {muxer_w}×{muxer_h}")
    print(f"nvstreammux 해상도: {muxer_w}×{muxer_h}")

    pipeline = Gst.Pipeline()

    # nvstreammux
    mux = Gst.ElementFactory.make("nvstreammux", "muxer")
    mux.set_property("width",                muxer_w)
    mux.set_property("height",               muxer_h)
    mux.set_property("batch-size",           NUM_SOURCES)
    mux.set_property("batched-push-timeout", 40000)
    pipeline.add(mux)

    for i, src in enumerate(VIDEO_SOURCES):
        make_src_and_connect(i, src, mux, pipeline)

    # nvinfer — C++ 파서(NvDsInferParseYolo11)로 탐지 결과를 NvDsObjectMeta에 기록
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
            tee    = Gst.ElementFactory.make("tee",      "tee")
            q_disp = Gst.ElementFactory.make("queue",    "q-disp")
            q_save = Gst.ElementFactory.make("queue",    "q-save")
            sink   = Gst.ElementFactory.make("nv3dsink", "sink")
            sink.set_property("sync", False)

            for el in (tee, q_disp, q_save, sink):
                pipeline.add(el)

            conv_out.link(tee)
            tee.request_pad_simple("src_%u").link(q_disp.get_static_pad("sink"))
            q_disp.link(sink)
            tee.request_pad_simple("src_%u").link(q_save.get_static_pad("sink"))
            link_save_branch(pipeline, q_save, OUTPUT_PATH)
            print(f"출력: nv3dsink {tiler_rows}×{tiler_cols} 타일 디스플레이 + 저장 → {OUTPUT_PATH}")

        elif USE_DISPLAY:
            sink = Gst.ElementFactory.make("nv3dsink", "sink")
            sink.set_property("sync", False)
            pipeline.add(sink)
            conv_out.link(sink)
            print(f"출력: nv3dsink {tiler_rows}×{tiler_cols} 타일 디스플레이")

        else:
            link_save_branch(pipeline, conv_out, OUTPUT_PATH)
            print(f"출력: 저장 전용 → {OUTPUT_PATH}")

    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        pgie.link(sink)
        print("출력: fakesink (헤드리스) — 화면 출력은 DISPLAY=:0, 저장은 --output 사용")

    # FPS 카운터 프로브 등록
    pgie_src = pgie.get_static_pad("src")
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, _fps_probe, 0)
    # obj_meta 진단 프로브 등록 (처음 5배치만 출력)
    pgie_src.add_probe(Gst.PadProbeType.BUFFER, _obj_meta_probe, 0)

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

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
