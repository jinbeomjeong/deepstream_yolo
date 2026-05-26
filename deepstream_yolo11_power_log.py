#!/usr/bin/env python3
"""
DeepStream 4채널 YOLO11m 추론 파이프라인 + 소비 전력 측정 (tegrastats, CSV 저장)

실행:
  source /home/nvidia/workspace/arround_view/venv/bin/activate
  python deepstream_yolo11_power_log.py [video1 .. video4]                    # 헤드리스
  python deepstream_yolo11_power_log.py --display [video1 ..]                 # 화면 출력
  python deepstream_yolo11_power_log.py --power-interval 1.0 --power-csv out.csv

측정 항목 (tegrastats):
  VDD_GPU_SOC  — GPU + SoC 전력 (mW)
  VDD_CPU_CV   — CPU + CV 전력 (mW)
  VIN_SYS_5V0  — 시스템 전체 전력 (5 V 레일, mW)
  VDDQ_VDD2_1V8AO — 메모리 전력 (mW)

CSV 컬럼:
  timestamp, elapsed_s,
  gpu_soc_mw, cpu_cv_mw, sys_5v0_mw, mem_mw, total_mw
"""

import sys
import os
import re
import csv
import time
import signal
import argparse
import threading
import subprocess
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from ds_pipeline import probe_video_size, bus_call, make_src_and_connect

try:
    import pyds
    _PYDS_OK = True
except ImportError:
    _PYDS_OK = False

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="DeepStream YOLO11m FP16 + 전력 로깅")
_ap.add_argument("sources", nargs="*", metavar="VIDEO",
                 help="입력 영상 경로 (최대 4개)")
_ap.add_argument("--display", "-d", action="store_true", default=False,
                 help="화면 출력 활성화 (기본값: 비활성화/헤드리스)")
_ap.add_argument("--power-interval", "-p", type=float, default=0.5, metavar="SEC",
                 help="전력 측정 주기 (초, 기본값: 0.5)")
_ap.add_argument("--power-csv", "-c", metavar="FILE", default=None,
                 help="전력 CSV 저장 경로 (기본값: power_YYYYMMDD_HHMMSS.csv)")
_args = _ap.parse_args()

_DEFAULT_SRC  = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
VIDEO_SOURCES  = _args.sources[:4] if _args.sources else [_DEFAULT_SRC] * 4
NUM_SOURCES    = len(VIDEO_SOURCES)
USE_DISPLAY    = _args.display
POWER_INTERVAL = max(0.1, _args.power_interval)
POWER_CSV      = _args.power_csv or f"power_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PGIE_CONFIG   = os.path.join(_PROJECT_ROOT, "config", "config_infer_yolo11_gpu_fp16.txt")

# ── 전력 로거 ─────────────────────────────────────────────────────────────────
# tegrastats 한 줄 예시:
#   ... VDD_GPU_SOC 5208mW/5208mW VDD_CPU_CV 2805mW/2805mW ...
# 패턴: NAME <cur>mW/<avg>mW  → cur(현재값) 사용
_POWER_RE = re.compile(
    r"VDD_GPU_SOC\s+(\d+)mW/\d+mW"
    r".*?VDD_CPU_CV\s+(\d+)mW/\d+mW"
    r".*?VIN_SYS_5V0\s+(\d+)mW/\d+mW"
    r".*?VDDQ_VDD2_1V8AO\s+(\d+)mW/\d+mW",
    re.DOTALL,
)

class PowerLogger:
    """
    별도 스레드에서 tegrastats를 실행하여 전력을 읽고 CSV에 기록한다.
    interval_sec : tegrastats 샘플링 주기 (초)
    csv_path     : 저장할 CSV 파일 경로
    """

    CSV_HEADER = [
        "timestamp", "elapsed_s",
        "gpu_soc_mw", "cpu_cv_mw", "sys_5v0_mw", "mem_mw", "total_mw",
    ]

    def __init__(self, interval_sec: float, csv_path: str):
        self._interval_ms = int(interval_sec * 1000)
        self._csv_path    = csv_path
        self._t0          = time.time()
        self._stop        = threading.Event()
        self._proc        = None
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[전력] 측정 시작 — 주기={self._interval_ms}ms, CSV={self._csv_path}")

    def stop(self):
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._thread.join(timeout=5)
        print(f"[전력] 측정 종료 — {self._csv_path} 저장 완료")

    def _run(self):
        cmd = ["tegrastats", "--interval", str(self._interval_ms)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            print("[전력] tegrastats 없음 — 전력 측정 비활성화")
            return

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                m = _POWER_RE.search(line)
                if not m:
                    continue
                gpu_soc = int(m.group(1))
                cpu_cv  = int(m.group(2))
                sys_5v0 = int(m.group(3))
                mem     = int(m.group(4))
                total   = gpu_soc + cpu_cv + mem  # 5V 레일은 중복 포함이므로 합산 제외
                elapsed = round(time.time() - self._t0, 3)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    elapsed,
                    gpu_soc, cpu_cv, sys_5v0, mem, total,
                ])
                f.flush()

        if self._proc.poll() is None:
            self._proc.terminate()

# ── 파이프라인 ────────────────────────────────────────────────────────────────
pipeline    = None
_pwr_logger = None

def main():
    global pipeline, _pwr_logger
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

    loop = GLib.MainLoop()
    bus  = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    def _graceful_stop(signum, frame):
        print("\n종료 신호 수신 — EOS 전송 중...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT,  _graceful_stop)

    for i, src in enumerate(VIDEO_SOURCES):
        print(f"소스[{i}]: {src}")
    print(f"YOLO11m GPU FP16 파이프라인 시작 ({NUM_SOURCES}채널, Ctrl+C 종료)...")

    _pwr_logger = PowerLogger(POWER_INTERVAL, POWER_CSV)
    _pwr_logger.start()

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()

    _pwr_logger.stop()
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
