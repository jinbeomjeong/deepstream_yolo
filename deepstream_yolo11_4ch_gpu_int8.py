#!/usr/bin/env python3

import sys
import os
import argparse
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import pyds

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepStream 4채널 YOLO11m INT8 추론")
    p.add_argument(
        "video", nargs="?",
        default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
        help="입력 영상 경로",
    )
    p.add_argument(
        "--sink", choices=("auto", "display", "fakesink"), default="auto",
        help="출력 sink 선택 (auto: DISPLAY 환경변수 따름, display: 화면 출력, fakesink: 출력 없음)",
    )
    return p.parse_args()


ARGS = parse_args()

# ── 설정 ──────────────────────────────────────────────────────────────────────
VIDEO_SOURCE = ARGS.video
PGIE_CONFIG = "/home/nvidia/workspace/deepstream_yolo/config/config_infer_yolo11_gpu_int8.txt"
NUM_SOURCES = 4
MUXER_W     = 1920
MUXER_H     = 1080
# --sink display → 강제 화면 출력, --sink fakesink → 강제 fakesink, auto → DISPLAY 따름
if ARGS.sink == "display":
    USE_DISPLAY = True
elif ARGS.sink == "fakesink":
    USE_DISPLAY = False
else:
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

    # nvinfer: INT8 추론 + C++ 파서 + NMS
    # labelfile-path → obj_label, text_params(위치·폰트 포함)를 자동 설정
    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pipeline.add(pgie)

    if USE_DISPLAY:
        tiler    = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
        conv_osd = Gst.ElementFactory.make("nvvideoconvert",     "conv-osd")
        # nvdsosd: display-text=1(기본) → text_params.display_text(클래스명) 렌더링
        osd      = Gst.ElementFactory.make("nvdsosd",            "osd")
        conv_out = Gst.ElementFactory.make("nvvideoconvert",     "conv-out")
        sink     = Gst.ElementFactory.make("nv3dsink",           "sink")

        tiler.set_property("rows",    2)
        tiler.set_property("columns", 2)
        tiler.set_property("width",   MUXER_W)
        tiler.set_property("height",  MUXER_H)
        osd.set_property("process-mode", 1)   # GPU 렌더링
        sink.set_property("sync", True)

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
