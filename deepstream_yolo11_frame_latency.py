#!/usr/bin/env python3
"""
DeepStream 4채널 YOLO11m 추론 파이프라인 + PGIE 배치 처리 시간 측정 (CSV 저장)

실행:
  source /home/nvidia/workspace/arround_view/venv/bin/activate
  python deepstream_yolo11_frame_latency.py [video1 .. video4]          # 헤드리스
  python deepstream_yolo11_frame_latency.py --display [video1 ..]       # 화면 출력
  python deepstream_yolo11_frame_latency.py --latency-csv out.csv       # CSV 경로 지정

측정 방법:
  nvinfer sink 패드 프로브에서 배치 시작 시각을 기록하고,
  nvinfer src 패드 프로브에서 같은 배치의 종료 시각과 비교해
  PGIE 배치 처리 시간(ms)을 기록한다.

CSV 컬럼:
  batch_seq      — 전체 누적 배치 번호
  timestamp      — 배치 도착 시각 (wall-clock)
  elapsed_s      — 파이프라인 시작 후 경과 시간 (초)
  batch_size     — 해당 배치의 프레임 수
  pgie_batch_ms  — PGIE sink→src 배치 처리 시간 (ms)
"""

import sys
import os
import csv
import time
import signal
import argparse
import threading
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from ds_pipeline import probe_video_size, bus_call, make_src_and_connect

try:
    import pyds
    _PYDS_OK = True
except ImportError:
    _PYDS_OK = False
    print("[경고] pyds 없음 — 프레임 타이밍 측정 불가 (venv 활성화 필요)")
    sys.exit(1)

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="DeepStream YOLO11m FP16 + PGIE 배치 처리 시간 측정")
_ap.add_argument("sources", nargs="*", metavar="VIDEO",
                 help="입력 영상 경로 (최대 4개)")
_ap.add_argument("--display", "-d", action="store_true", default=False,
                 help="화면 출력 활성화 (기본값: 비활성화/헤드리스)")
_ap.add_argument("--latency-csv", "-c", metavar="FILE", default=None,
                 help="PGIE 배치 처리 시간 CSV 저장 경로 "
                      "(기본값: frame_latency_YYYYMMDD_HHMMSS.csv)")
_args = _ap.parse_args()

_DEFAULT_SRC  = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
VIDEO_SOURCES = _args.sources[:4] if _args.sources else [_DEFAULT_SRC] * 4
NUM_SOURCES   = len(VIDEO_SOURCES)
USE_DISPLAY   = _args.display
LATENCY_CSV   = (_args.latency_csv
                 or f"frame_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PGIE_CONFIG   = os.path.join(_PROJECT_ROOT, "config", "config_infer_yolo11_gpu_fp16.txt")

# ── 프레임 타이밍 측정 상태 ────────────────────────────────────────────────────
_t_start      = time.perf_counter()          # 파이프라인 기준 시각

# PGIE 진입 시각
_pending_batch_start_t = []
_latency_lock  = threading.Lock()
_batch_seq     = 0                            # 전체 누적 배치 카운터

# CSV 쓰기용 큐 (별도 스레드로 IO 분리)
_csv_queue    = []
_csv_lock     = threading.Lock()
_csv_stop     = threading.Event()

CSV_HEADER = [
    "batch_seq", "timestamp", "elapsed_s",
    "batch_size", "pgie_batch_ms",
]

def _csv_writer_thread(path: str):
    """백그라운드에서 CSV 행을 기록한다."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        while not _csv_stop.is_set() or _csv_queue:
            with _csv_lock:
                rows, _csv_queue[:] = _csv_queue[:], []
            for row in rows:
                writer.writerow(row)
            if rows:
                f.flush()
            else:
                time.sleep(0.005)

# ── FPS 출력 (콘솔 전용) ──────────────────────────────────────────────────────
_fps_frame_count = 0
_fps_win_t       = time.perf_counter()
_fps_win_frames  = 0

# ── nvinfer 패드 프로브 ────────────────────────────────────────────────────────
def _pgie_sink_probe(pad, info, u_data):
    """PGIE 입력 시점에 배치 시작 시각을 저장한다."""
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    with _latency_lock:
        _pending_batch_start_t.append(time.perf_counter())

    return Gst.PadProbeReturn.OK


def _latency_probe(pad, info, u_data):
    """
    배치 단위로 호출된다.
    PGIE sink 진입 시각과 PGIE src 도착 시각의 차이를 CSV 큐에 추가한다.
    """
    global _batch_seq, _fps_frame_count, _fps_win_t, _fps_win_frames

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    now      = time.perf_counter()
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    elapsed  = round(now - _t_start, 6)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    batch_size = 0
    l_frame  = batch_meta.frame_meta_list
    while l_frame:
        try:
            pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        batch_size += 1
        _fps_frame_count  += 1
        _fps_win_frames   += 1

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    with _latency_lock:
        start_t = _pending_batch_start_t.pop(0) if _pending_batch_start_t else None

    pgie_batch_ms = round((now - start_t) * 1000, 3) if start_t is not None else 0.0
    _batch_seq += 1

    # CSV 큐에 추가
    with _csv_lock:
        _csv_queue.append([
            _batch_seq, now_str, elapsed,
            batch_size, pgie_batch_ms,
        ])

    # 100프레임마다 FPS 출력
    if _fps_frame_count % 100 == 0:
        now2    = time.perf_counter()
        elapsed2 = now2 - _fps_win_t
        fps     = _fps_win_frames / elapsed2 if elapsed2 > 0 else 0.0
        print(f"[{_fps_frame_count:6d}프레임] FPS={fps:.1f}")
        _fps_win_t      = now2
        _fps_win_frames = 0

    return Gst.PadProbeReturn.OK

# ── 파이프라인 ────────────────────────────────────────────────────────────────
pipeline = None

def main():
    global pipeline, _t_start, _fps_win_t
    Gst.init(None)

    muxer_w, muxer_h = probe_video_size(VIDEO_SOURCES[0])
    if not (muxer_w and muxer_h):
        muxer_w, muxer_h = 1920, 1080
        print(f"[경고] 해상도 탐색 실패 — 기본값 사용: {muxer_w}×{muxer_h}")
    print(f"nvstreammux 해상도: {muxer_w}×{muxer_h}")

    pipeline = Gst.Pipeline()

    mux = Gst.ElementFactory.make("nvstreammux", "muxer")
    mux.set_property("width",                muxer_w)
    mux.set_property("height",               muxer_h)
    mux.set_property("batch-size",           NUM_SOURCES)
    mux.set_property("batched-push-timeout", 40000)
    pipeline.add(mux)

    for i, src in enumerate(VIDEO_SOURCES):
        make_src_and_connect(i, src, mux, pipeline)

    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pipeline.add(pgie)
    mux.link(pgie)

    if USE_DISPLAY:
        tiler    = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
        conv_osd = Gst.ElementFactory.make("nvvideoconvert",     "conv-osd")
        osd      = Gst.ElementFactory.make("nvdsosd",            "osd")
        conv_out = Gst.ElementFactory.make("nvvideoconvert",     "conv-out")
        sink     = Gst.ElementFactory.make("nv3dsink",           "sink")

        tiler_cols = min(NUM_SOURCES, 2)
        tiler_rows = (NUM_SOURCES + tiler_cols - 1) // tiler_cols
        tiler.set_property("rows",    tiler_rows)
        tiler.set_property("columns", tiler_cols)
        tiler.set_property("width",   1920)
        tiler.set_property("height",  1080)
        osd.set_property("process-mode", 1)
        sink.set_property("sync", False)

        for el in (tiler, conv_osd, osd, conv_out, sink):
            pipeline.add(el)

        pgie.link(tiler)
        tiler.link(conv_osd)
        conv_osd.link(osd)
        osd.link(conv_out)
        conv_out.link(sink)
        print(f"출력: nv3dsink {tiler_rows}×{tiler_cols} 타일")

    else:
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        pipeline.add(sink)
        pgie.link(sink)
        print("출력: fakesink (헤드리스 — 화면 출력은 --display 사용)")

    # 프레임 타이밍 프로브 등록
    pgie.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, _pgie_sink_probe, 0)
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, _latency_probe, 0)

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def _graceful_stop(signum, frame):
        print("\n종료 신호 수신 — EOS 전송 중...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT,  _graceful_stop)

    # CSV 백그라운드 writer 시작
    writer_thread = threading.Thread(
        target=_csv_writer_thread, args=(LATENCY_CSV,), daemon=True
    )
    writer_thread.start()
    print(f"[타이밍] CSV 저장 시작 → {LATENCY_CSV}")

    for i, src in enumerate(VIDEO_SOURCES):
        print(f"소스[{i}]: {src}")
    print(f"YOLO11m GPU FP16 파이프라인 시작 ({NUM_SOURCES}채널, Ctrl+C 종료)...")

    _t_start = time.perf_counter()
    _fps_win_t = _t_start
    pipeline.set_state(Gst.State.PLAYING)
    loop.run()

    # CSV writer 종료 대기
    _csv_stop.set()
    writer_thread.join(timeout=5)
    print(f"[타이밍] CSV 저장 완료 → {LATENCY_CSV}  (총 {_batch_seq}배치)")

    elapsed_total = time.perf_counter() - _t_start
    fps_avg = _fps_frame_count / elapsed_total if elapsed_total > 0 else 0
    print(f"총 처리: {_fps_frame_count}프레임 | 평균 FPS={fps_avg:.1f}")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
