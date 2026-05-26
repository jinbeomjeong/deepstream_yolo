import os
import sys

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GstPbutils


def probe_video_size(path):
    """GstPbutils.Discoverer로 영상의 실제 해상도(w, h)를 반환. 실패 시 (None, None)."""
    uri = f"file://{os.path.abspath(path)}"
    try:
        disc = GstPbutils.Discoverer.new(5 * Gst.SECOND)
        info = disc.discover_uri(uri)
        for stream in info.get_video_streams():
            return stream.get_width(), stream.get_height()
    except Exception as e:
        print(f"[경고] 해상도 탐색 실패 ({path}): {e}")
    return None, None


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


def make_src_and_connect(idx, path, mux, pipeline):
    """uridecodebin 소스를 생성하고 nvstreammux 의 sink_{idx} 패드에 연결한다."""
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


def link_save_branch(pipeline, src_el, output_path):
    """src_el 뒤에 nvv4l2h264enc HW 인코더 → mp4 저장 브랜치를 연결한다."""
    conv_enc  = Gst.ElementFactory.make("nvvideoconvert", "conv-enc")
    encoder   = Gst.ElementFactory.make("nvv4l2h264enc",  "encoder")
    h264parse = Gst.ElementFactory.make("h264parse",      "h264-parse")
    mux_mp4   = Gst.ElementFactory.make("mp4mux",         "mp4-mux")
    filesink  = Gst.ElementFactory.make("filesink",       "filesink")

    encoder.set_property("bitrate",      8000000)   # 8 Mbps
    encoder.set_property("preset-level", 1)          # UltraFast (HW 최고속)
    encoder.set_property("idrinterval",  60)          # IDR 60프레임(2초) — 표준 GOP
    filesink.set_property("location",    output_path)
    filesink.set_property("sync",        False)

    for el in (conv_enc, encoder, h264parse, mux_mp4, filesink):
        pipeline.add(el)

    src_el.link(conv_enc)
    conv_enc.link(encoder)
    encoder.link(h264parse)
    h264parse.link(mux_mp4)
    mux_mp4.link(filesink)
