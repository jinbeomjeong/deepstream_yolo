#!/usr/bin/env python3
"""
DeepStream 4채널 YOLO11m DLA INT8 추론 파이프라인

추론 흐름:
  nvinfer (DLA core 0 INT8 엔진, GPU fallback)
    → lib_parser_yolo.so NvDsInferParseYolo11  [84,8400] 텐서 파싱
    → DeepStream NMS (cluster-mode=2)
    → NvDsObjectMeta (좌표 역변환·obj_label·text_params를 nvinfer가 자동 설정)
    → nvdsosd         (클래스명 텍스트·바운딩박스 렌더링, probe 불필요)

실행:
  python3 deepstream_yolo11_4ch_dla_int8.py [video_path]
  DISPLAY=:0 python3 deepstream_yolo11_4ch_dla_int8.py [video_path]
"""

import sys
import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import pyds

# ── 설정 ──────────────────────────────────────────────────────────────────────
VIDEO_SOURCE = (
    sys.argv[1] if len(sys.argv) > 1
    else "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
)
PGIE_CONFIG = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolo11_dla_int8.txt"
NUM_SOURCES = 4
MUXER_W     = 1920
MUXER_H     = 1080
USE_DISPLAY = bool(os.environ.get("DISPLAY"))


# ── 버스 콜백 ──────────────────────────────────────────────────────────────────
pipeline = None

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("EOS — 영상 종료")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"ERROR: {err}: {debug}\n")
        loop.quit()
    return True


# ── 소스 연결 ──────────────────────────────────────────────────────────────────
def make_src_and_connect(idx, path, mux, pipeline):
    udbin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-{idx}")
    udbin.set_property("uri", f"file://{path}")
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


# ── 파이프라인 ─────────────────────────────────────────────────────────────────
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

    for i in range(NUM_SOURCES):
        make_src_and_connect(i, VIDEO_SOURCE, mux, pipeline)

    # nvinfer: DLA core 0 INT8 추론 + C++ 파서 + NMS
    # DLA 미지원 레이어(attention 등)는 GPU 자동 폴백
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pipeline.add(pgie)

    if USE_DISPLAY:
        tiler    = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
        conv_osd = Gst.ElementFactory.make("nvvideoconvert",     "conv-osd")
        osd      = Gst.ElementFactory.make("nvdsosd",            "osd")
        conv_out = Gst.ElementFactory.make("nvvideoconvert",     "conv-out")
        sink     = Gst.ElementFactory.make("nv3dsink",           "sink")

        tiler.set_property("rows",    2)
        tiler.set_property("columns", 2)
        tiler.set_property("width",   MUXER_W)
        tiler.set_property("height",  MUXER_H)
        osd.set_property("process-mode", 1)
        sink.set_property("sync", False)

        for el in (tiler, conv_osd, osd, conv_out, sink):
            pipeline.add(el)

        mux.link(pgie)
        pgie.link(tiler)
        tiler.link(conv_osd)
        conv_osd.link(osd)
        osd.link(conv_out)
        conv_out.link(sink)
    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        mux.link(pgie)
        pgie.link(sink)

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
