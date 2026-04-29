#!/usr/bin/env python3
"""
deepstream_yolov8_4ch_dla.py
============================
4채널 영상을 처음부터 2채널씩 분리하여 각 DLA 코어에 직접 연결하는
최적화된 DeepStream 파이프라인

[이전 구조 — 비효율]
  uridecodebin×4 → nvstreammux(4) → nvstreamdemux
                                   → nvstreammux_dla0(2) → nvinfer_dla0
                                   → nvstreammux_dla1(2) → nvinfer_dla1

[개선된 구조 — 직결]
  uridecodebin(ch0) ─┐
  uridecodebin(ch1) ─┴→ nvstreammux_dla0(batch=2) → nvinfer_dla0 ─┐
                                                                    funnel → tiler → osd → sink
  uridecodebin(ch2) ─┐                                             │
  uridecodebin(ch3) ─┴→ nvstreammux_dla1(batch=2) → nvinfer_dla1 ─┘

[제거된 엘리먼트]
  - nvstreammux_main  (4채널 통합 mux → 불필요)
  - nvstreamdemux     (분리 과정 → 불필요)
  → GPU/CPU 메모리 복사 1회 감소, 동기화 오버헤드 제거

[실행]
  python3 deepstream_yolov8_4ch_dla.py [video_path]
  python3 deepstream_yolov8_4ch_dla.py video.mp4 --display
"""

import sys
import os
import time
import logging
import argparse
import gi
try:
    import nvtx
    _NVTX = True
except ImportError:
    import contextlib
    class _NvtxStub:
        @staticmethod
        def annotate(msg="", color=None, **kw):
            return contextlib.nullcontext()
    nvtx = _NvtxStub()
    _NVTX = False

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import pyds

# ══════════════════════════════════════════════════════════════════════════════
# 로거 설정
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("deepstream_yolo")


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
PGIE_CONFIG_DLA0 = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolov8_dla0_int8.txt"
PGIE_CONFIG_DLA1 = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolov8_dla1_int8.txt"

MUXER_W   = 1920
MUXER_H   = 1080
DLA_BATCH = 2          # DLA 코어 1개당 채널 수

# DLA 코어별 담당 채널 정의
# [비유] 처음부터 "채널 0,1은 계산대 A, 채널 2,3은 계산대 B"로 배정
DLA_GROUPS = {
    0: [0, 1],   # DLA Core 0 → 채널 0, 1
    1: [2, 3],   # DLA Core 1 → 채널 2, 3
}

FPS_REPORT_INTERVAL = 2.0   # 초 단위 FPS 출력 주기


# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="DeepStream 4채널 YOLOv8m 2×DLA 파이프라인")
    parser.add_argument(
        "video", nargs="?",
        default="/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
        help="입력 영상 경로 (기본값: DeepStream 샘플 영상)",
    )
    parser.add_argument(
        "--display", action="store_true", default=bool(os.environ.get("DISPLAY")),
        help="화면 출력 활성화 (기본: DISPLAY 환경변수 자동 감지)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# FPS 카운터
# ══════════════════════════════════════════════════════════════════════════════
class FpsCounter:
    """채널별 독립 FPS 측정기."""

    def __init__(self, n_channels: int):
        self._cnt = [0]   * n_channels
        self._t0  = [time.perf_counter()] * n_channels
        self._fps = [0.0] * n_channels

    def tick(self, ch_id: int) -> bool:
        """프레임 1개 카운트. FPS_REPORT_INTERVAL 경과 시 FPS 갱신 후 True 반환."""
        self._cnt[ch_id] += 1
        now     = time.perf_counter()
        elapsed = now - self._t0[ch_id]
        if elapsed >= FPS_REPORT_INTERVAL:
            self._fps[ch_id] = self._cnt[ch_id] / elapsed
            self._cnt[ch_id] = 0
            self._t0[ch_id]  = now
            return True
        return False

    def get(self, ch_id: int) -> float:
        return self._fps[ch_id]

    def all(self) -> list:
        return self._fps[:]


_fps_counter = FpsCounter(n_channels=4)


# ══════════════════════════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def make_element(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(
            f"엘리먼트 생성 실패: '{factory}' (name='{name}')\n"
            f"  플러그인 확인: gst-inspect-1.0 {factory}"
        )
    return el


# ══════════════════════════════════════════════════════════════════════════════
# 버스 콜백
# ══════════════════════════════════════════════════════════════════════════════
def bus_call(bus, message, loop: GLib.MainLoop) -> bool:
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.info("EOS — 모든 채널 영상 종료")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        logger.warning("%s: %s", warn, debug)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("%s: %s", err, debug)
        loop.quit()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 소스 → nvstreammux_dlaX 직결
# ══════════════════════════════════════════════════════════════════════════════
def add_source_to_mux(pipeline: Gst.Pipeline,
                      ch_idx: int,
                      uri: str,
                      mux: Gst.Element,
                      mux_sink_idx: int) -> None:
    """
    채널 ch_idx의 uridecodebin을 생성하고 지정된 mux의 sink_N에 직접 연결.

    [핵심 변경]
    이전: uridecodebin → mux_main → demux → mux_dlaX
    이번: uridecodebin ───────────────────→ mux_dlaX  (중간 단계 완전 제거)

    Parameters
    ----------
    ch_idx       : 전체 채널 번호 (0~3), 소스 고유 식별자
    mux          : 연결할 nvstreammux_dlaX
    mux_sink_idx : 해당 mux 내에서의 sink 패드 번호 (전체 채널 번호와 일치)
    """
    uri_full = uri if uri.startswith("file://") else f"file://{os.path.abspath(uri)}"
    udbin = make_element("uridecodebin", f"udbin-ch{ch_idx}")
    udbin.set_property("uri", uri_full)
    pipeline.add(udbin)

    sink_pad = mux.request_pad_simple(f"sink_{mux_sink_idx}")
    if not sink_pad:
        raise RuntimeError(
            f"mux sink_{mux_sink_idx} 패드 요청 실패 "
            f"(채널 {ch_idx})"
        )

    def on_pad_added(dec, pad, sink_pad, ch):
        caps     = pad.get_current_caps()
        gstname  = caps.get_structure(0).get_name() if caps else ""
        features = caps.get_features(0) if caps else None
        if "video" in gstname and features and features.contains("memory:NVMM"):
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                status = "OK" if ret == Gst.PadLinkReturn.OK else f"실패({ret})"
                logger.info("채널 %d → %s 연결 %s", ch, sink_pad.get_parent_element().get_name(), status)

    udbin.connect("pad-added", on_pad_added, sink_pad, ch_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 파이프라인 구성
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(video_source: str, use_display: bool) -> tuple:
    """
    최적화된 파이프라인 조립.

    [조립 순서]
    1. DLA 코어별로 nvstreammux + nvinfer 쌍을 생성
    2. 각 채널의 uridecodebin을 해당 mux에 직접 연결
    3. 두 nvinfer 출력을 funnel로 합류
    4. 후처리(tiler, osd) → sink
    """
    with nvtx.annotate("pipeline_init", color="green"):
        Gst.init(None)
    pipeline = Gst.Pipeline()
    loop     = GLib.MainLoop()

    pgie_configs = {0: PGIE_CONFIG_DLA0, 1: PGIE_CONFIG_DLA1}
    nvinfer_outputs = []   # funnel 연결을 위해 nvinfer 엘리먼트 보관

    # ── 1. DLA 코어별 (mux → nvinfer) 쌍 구성 ────────────────────────────
    with nvtx.annotate("build_dla_pairs", color="blue"):
        for dla_core, channels in DLA_GROUPS.items():
            logger.info("DLA Core %d 구성 — 채널 %s", dla_core, channels)

            # nvstreammux_dlaX
            mux = make_element("nvstreammux", f"mux-dla{dla_core}")
            mux.set_property("width",                MUXER_W)
            mux.set_property("height",               MUXER_H)
            mux.set_property("batch-size",           DLA_BATCH)
            mux.set_property("batched-push-timeout", 40_000)   # μs
            mux.set_property("nvbuf-memory-type",    4)        # NVBUF_MEM_SURFACE_ARRAY: Jetson 유일 지원 타입
            pipeline.add(mux)

            # 채널 소스 직결
            for ch_idx in channels:
                add_source_to_mux(pipeline, ch_idx, video_source, mux, ch_idx)

            # nvinfer_dlaX
            pgie = make_element("nvinfer", f"pgie-dla{dla_core}")
            pgie.set_property("config-file-path", pgie_configs[dla_core])
            pipeline.add(pgie)

            if not mux.link(pgie):
                raise RuntimeError(f"mux-dla{dla_core} → pgie-dla{dla_core} 링크 실패")

            nvinfer_outputs.append(pgie)
            logger.info("mux-dla%d → pgie-dla%d 연결 완료 (설정: %s)",
                        dla_core, dla_core, pgie_configs[dla_core])

    # ── 2. funnel (두 nvinfer 출력 합류) ───────────────────────────────
    funnel = make_element("funnel", "funnel")
    pipeline.add(funnel)

    for pgie in nvinfer_outputs:
        if not pgie.link(funnel):
            raise RuntimeError(f"{pgie.get_name()} → funnel 링크 실패")
        logger.info("%s → funnel 연결 완료", pgie.get_name())

    # ── 3. 후처리 + 출력 ─────────────────────────────────────────────────
    if use_display:
        tiler    = make_element("nvmultistreamtiler", "tiler")
        conv_osd = make_element("nvvideoconvert",     "conv-osd")
        osd      = make_element("nvdsosd",            "osd")
        conv_out = make_element("nvvideoconvert",     "conv-out")
        sink     = make_element("nv3dsink",           "sink")

        # 2×2 타일: 채널 0,1 (상단) / 채널 2,3 (하단)
        tiler.set_property("rows",    2)
        tiler.set_property("columns", 2)
        tiler.set_property("width",   MUXER_W)
        tiler.set_property("height",  MUXER_H)
        osd.set_property("process-mode", 1)   # GPU 렌더링
        sink.set_property("sync", False)

        for el in (tiler, conv_osd, osd, conv_out, sink):
            pipeline.add(el)

        for src, dst in [(funnel, tiler), (tiler, conv_osd),
                         (conv_osd, osd), (osd, conv_out), (conv_out, sink)]:
            if not src.link(dst):
                raise RuntimeError(
                    f"{src.get_name()} → {dst.get_name()} 링크 실패"
                )

        logger.info("디스플레이 출력 모드: nv3dsink (2×2 타일)")
    else:
        sink = make_element("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        if not funnel.link(sink):
            raise RuntimeError("funnel → fakesink 링크 실패")
        logger.info("헤드리스 모드: fakesink")

    return pipeline, loop


# ══════════════════════════════════════════════════════════════════════════════
# 추론 결과 probe
# ══════════════════════════════════════════════════════════════════════════════
def attach_inference_probe(pipeline: Gst.Pipeline) -> None:
    """
    각 nvinfer 출력 패드에 probe를 부착하여 탐지 결과를 콘솔에 출력.
    성능 측정이나 디버깅 시 활용. 불필요 시 main()에서 호출 제거.
    """
    _PROBE_COLORS = {0: "yellow", 1: "cyan"}

    def make_probe(dla_core: int):
        _color = _PROBE_COLORS.get(dla_core, "white")
        def probe_cb(pad, info):
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK

            with nvtx.annotate(f"infer_probe_dla{dla_core}", color=_color):
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                frame_list = batch_meta.frame_meta_list
                while frame_list:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    except StopIteration:
                        break

                    n_objs = 0
                    obj_list = frame_meta.obj_meta_list
                    while obj_list:
                        try:
                            pyds.NvDsObjectMeta.cast(obj_list.data)
                            n_objs += 1
                            obj_list = obj_list.next
                        except StopIteration:
                            break

                    logger.info("DLA%d ch=%d frame=%d objs=%d",
                                dla_core, frame_meta.source_id,
                                frame_meta.frame_num, n_objs)
                    try:
                        frame_list = frame_list.next
                    except StopIteration:
                        break

            return Gst.PadProbeReturn.OK
        return probe_cb

    for core in DLA_GROUPS:
        pgie = pipeline.get_by_name(f"pgie-dla{core}")
        if pgie:
            src_pad = pgie.get_static_pad("src")
            src_pad.add_probe(Gst.PadProbeType.BUFFER, make_probe(core))
            logger.info("pgie-dla%d probe 부착 완료", core)


# ══════════════════════════════════════════════════════════════════════════════
# FPS probe
# ══════════════════════════════════════════════════════════════════════════════
def attach_fps_probe(pipeline: Gst.Pipeline) -> None:
    """각 nvinfer src pad에 probe를 부착하여 채널별 FPS를 주기적으로 출력."""

    _FPS_COLORS = {0: "yellow", 1: "cyan"}

    def make_probe(dla_core: int):
        _color = _FPS_COLORS.get(dla_core, "white")
        def probe_cb(pad, info):
            buf = info.get_buffer()
            if not buf:
                return Gst.PadProbeReturn.OK

            with nvtx.annotate(f"fps_probe_dla{dla_core}", color=_color):
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                if not batch_meta:
                    return Gst.PadProbeReturn.OK

                updated = False
                frame_list = batch_meta.frame_meta_list
                while frame_list:
                    try:
                        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
                    except StopIteration:
                        break
                    if _fps_counter.tick(frame_meta.source_id):
                        updated = True
                    try:
                        frame_list = frame_list.next
                    except StopIteration:
                        break

                if updated:
                    fps = _fps_counter.all()
                    logger.info(
                        "FPS | ch0=%5.1f  ch1=%5.1f  ch2=%5.1f  ch3=%5.1f",
                        fps[0], fps[1], fps[2], fps[3],
                    )
            return Gst.PadProbeReturn.OK
        return probe_cb

    for core in DLA_GROUPS:
        pgie = pipeline.get_by_name(f"pgie-dla{core}")
        if pgie:
            pgie.get_static_pad("src").add_probe(
                Gst.PadProbeType.BUFFER, make_probe(core)
            )
    logger.info("FPS probe 부착 완료 (출력 주기: %.1fs)", FPS_REPORT_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeepStream 4채널 YOLOv8m — 2× DLA 직결 파이프라인")
    logger.info("=" * 60)
    logger.info("영상 소스  : %s", args.video)
    for core, channels in DLA_GROUPS.items():
        logger.info("DLA Core %d : 채널 %s", core, channels)
    logger.info("디스플레이 : %s", "활성 (nv3dsink)" if args.display else "비활성 (fakesink)")

    with nvtx.annotate("build_pipeline", color="green"):
        pipeline, loop = build_pipeline(args.video, args.display)
    attach_fps_probe(pipeline)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logger.info("파이프라인 시작...")
    with nvtx.annotate("pipeline_state_playing", color="orange"):
        ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("파이프라인 PLAYING 전환 실패")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(1)

    try:
        with nvtx.annotate("pipeline_running", color="purple"):
            loop.run()
    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C)")
    finally:
        fps = _fps_counter.all()
        logger.info(
            "최종 FPS | ch0=%5.1f  ch1=%5.1f  ch2=%5.1f  ch3=%5.1f",
            fps[0], fps[1], fps[2], fps[3],
        )
        logger.info("파이프라인 종료 중...")
        pipeline.set_state(Gst.State.NULL)
        logger.info("완료")


if __name__ == "__main__":
    main()
