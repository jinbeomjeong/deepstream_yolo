#!/usr/bin/env python3
"""
measure_video.py
================
deepstream_yolov8_video.py 실행 중 성능 측정 프로그램

측정 항목:
  - FPS          : deepstream_yolov8_video.py pad probe 기반 1초 윈도우 파싱
  - CPU 사용률   : 전체 코어 평균 (%)
  - GPU 사용률   : GR3D_FREQ (%)
  - 소비 전력    : VDD_CPU_CV / VDD_GPU_SOC / VDDQ_VDD2_1V8AO / VIN_SYS_5V0 / 합산 (mW)

출력:
  - CSV : measure_YYYYMMDD_HHMMSS.csv  (tegrastats 샘플 단위)
  - 콘솔 요약 통계

실행:
  python measure_video.py                        # DLA Core 0 INT8 (기본)
  python measure_video.py --dla-core 1           # DLA Core 1 INT8
  python measure_video.py --dla-core -1          # GPU INT8
  python measure_video.py --interval 1.0         # 측정 주기 1초
  python measure_video.py --output my.csv        # 출력 파일 지정
  python measure_video.py --no-display           # 디스플레이 없이 실행
"""

import os
import re
import csv
import sys
import time
import argparse
import threading
import subprocess
from datetime import datetime

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR  = "/home/nvidia/workspace/deepstream_yolo"
DS_SCRIPT = os.path.join(BASE_DIR, "deepstream_yolov8_video.py")
VENV_PY   = "/home/nvidia/workspace/arround_view/venv/bin/python"

# ── tegrastats 파싱 정규식 ────────────────────────────────────────────────────
_CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
_GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
_PWR_RE = {
    "vdd_cpu_cv_mw":      re.compile(r"VDD_CPU_CV (\d+)mW"),
    "vdd_gpu_soc_mw":     re.compile(r"VDD_GPU_SOC (\d+)mW"),
    "vddq_vdd2_1v8ao_mw": re.compile(r"VDDQ_VDD2_1V8AO (\d+)mW"),
    "vin_sys_5v0_mw":     re.compile(r"VIN_SYS_5V0 (\d+)mW"),
}

# pad probe 기반 FPS 메트릭: "[FPS] elapsed_s fps"
_FPS_RE = re.compile(r"\[FPS\] ([\d.]+) ([\d.]+)")

CSV_HEADER = [
    "timestamp", "elapsed_s",
    "fps_window",
    "cpu_avg_pct", "gpu_pct",
    "vdd_cpu_cv_mw", "vdd_gpu_soc_mw", "vddq_vdd2_1v8ao_mw", "vin_sys_5v0_mw",
    "all_mw",
]

def parse_args():
    p = argparse.ArgumentParser(description="deepstream_yolov8_video.py 성능 측정")
    p.add_argument("--video",      default=os.path.join(BASE_DIR, "video_h264.mp4"),
                   metavar="PATH", help="입력 비디오 파일")
    p.add_argument("--dla-core",   type=int, default=0,
                   help="가속기: 0=DLA Core 0 (기본), 1=DLA Core 1, -1=GPU INT8")
    p.add_argument("--no-display", action="store_true",
                   help="디스플레이 비활성화")
    p.add_argument("--interval",   type=float, default=0.5, metavar="SEC",
                   help="tegrastats 샘플링 주기 (초, 기본: 0.5)")
    p.add_argument("--output",      default=None, metavar="PATH",
                   help="출력 CSV 경로 (기본: measure_YYYYMMDD_HHMMSS.csv)")
    return p.parse_args()


def parse_tegrastats(line):
    """tegrastats 한 줄 → (cpu_avg_pct, gpu_pct, power_dict) 또는 None."""
    cpu_m = _CPU_RE.search(line)
    gpu_m = _GPU_RE.search(line)
    if not cpu_m or not gpu_m:
        return None

    cores = [int(s.split("%")[0]) for s in cpu_m.group(1).split(",") if "%" in s]
    cpu_avg = sum(cores) / len(cores) if cores else 0.0
    gpu_pct = int(gpu_m.group(1))

    power = {}
    for key, pat in _PWR_RE.items():
        m = pat.search(line)
        if not m:
            return None
        power[key] = int(m.group(1))
    power["all_mw"] = sum(power.values())

    return round(cpu_avg, 1), gpu_pct, power


class Measurement:
    """스레드 간 공유 상태."""

    def __init__(self):
        self._lock       = threading.Lock()
        self.fps         = None   # deepstream에서 파싱한 최신 구간 FPS
        self.rows        = []     # CSV 행 누적
        self.ds_done     = threading.Event()
        self._fps_windows: list = []  # [(start_elapsed_s, end_elapsed_s, fps)]

    def log_fps_window(self, start_e: float, end_e: float, fps: float):
        with self._lock:
            self._fps_windows.append((start_e, end_e, fps))
            self.fps = fps

    def get_fps(self):
        with self._lock:
            return self.fps

    def get_fps_windows(self):
        with self._lock:
            return list(self._fps_windows)

    def add_row(self, row: dict):
        with self._lock:
            self.rows.append(row)


# ── tegrastats 수집 스레드 ────────────────────────────────────────────────────
def tegrastats_thread(m: Measurement, interval_ms: int, t0: float):
    cmd = ["tegrastats", "--interval", str(interval_ms)]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
    except FileNotFoundError:
        print("[측정] tegrastats 없음 — CPU/GPU/전력 측정 불가")
        return

    try:
        for line in proc.stdout:
            if m.ds_done.is_set():
                break
            parsed = parse_tegrastats(line)
            if parsed is None:
                continue
            cpu_avg, gpu_pct, power = parsed
            now = time.time()
            m.add_row({
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "elapsed_s":          round(now - t0, 3),
                "fps_window":         m.get_fps(),
                "cpu_avg_pct":        cpu_avg,
                "gpu_pct":            gpu_pct,
                "vdd_cpu_cv_mw":      power["vdd_cpu_cv_mw"],
                "vdd_gpu_soc_mw":     power["vdd_gpu_soc_mw"],
                "vddq_vdd2_1v8ao_mw": power["vddq_vdd2_1v8ao_mw"],
                "vin_sys_5v0_mw":     power["vin_sys_5v0_mw"],
                "all_mw":             power["all_mw"],
            })
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


# ── deepstream 실행 스레드 ────────────────────────────────────────────────────
def deepstream_thread(m: Measurement, cmd: list, t0: float):
    """deepstream_yolov8_video.py 실행 + FPS 윈도우 파싱."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except Exception as e:
        print(f"[측정] deepstream 실행 실패: {e}")
        m.ds_done.set()
        return

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        fps_m = _FPS_RE.search(line)
        if fps_m:
            fps_val = float(fps_m.group(2))
            end_e   = round(time.time() - t0, 3)   # tegrastats와 동일한 기준
            start_e = round(end_e - 1.0, 3)
            m.log_fps_window(max(0.0, start_e), end_e, fps_val)

    proc.wait()
    m.ds_done.set()


# ── CSV 저장 ──────────────────────────────────────────────────────────────────
def save_csv(rows: list, path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[측정] CSV 저장: {path}  ({len(rows)}행)")


# ── 요약 통계 출력 ────────────────────────────────────────────────────────────
def print_summary(rows: list):
    if not rows:
        print("[측정] 수집된 데이터 없음")
        return

    def stats(vals, unit="", fmt=".1f"):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "N/A"
        return (f"avg={sum(vals)/len(vals):{fmt}}  "
                f"min={min(vals):{fmt}}  max={max(vals):{fmt}}{unit}")

    fps_v = [r["fps_window"]     for r in rows]
    cpu_v = [r["cpu_avg_pct"]    for r in rows]
    gpu_v = [r["gpu_pct"]        for r in rows]
    pwr_v = [r["all_mw"]/1000    for r in rows]
    cmw_v = [r["vdd_cpu_cv_mw"]/1000  for r in rows]
    gmw_v = [r["vdd_gpu_soc_mw"]/1000 for r in rows]

    elapsed = rows[-1]["elapsed_s"]
    print("\n" + "═" * 62)
    print("  성능 측정 요약")
    print("═" * 62)
    print(f"  전체 시간  : {elapsed:.1f} s  ({len(rows)} 샘플)")
    print("─" * 62)
    print(f"  FPS        : {stats(fps_v, ' fps')}")
    print(f"  CPU 사용률 : {stats(cpu_v, ' %')}")
    print(f"  GPU 사용률 : {stats(gpu_v, ' %')}")
    print(f"  전력 합산  : {stats(pwr_v, ' W')}")
    print(f"  CPU 전력   : {stats(cmw_v, ' W')}  (VDD_CPU_CV)")
    print(f"  GPU 전력   : {stats(gmw_v, ' W')}  (VDD_GPU_SOC)")
    print("\n" + "═" * 62)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_csv = args.output or \
        f"measure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    interval_ms = max(100, int(args.interval * 1000))

    # deepstream 실행 명령 구성
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    ds_cmd = [
        py, DS_SCRIPT,
        "--video",    args.video,
        "--dla-core", str(args.dla_core),
    ]
    if args.no_display:
        ds_cmd.append("--no-display")

    accel = (f"DLA Core {args.dla_core}" if args.dla_core in (0, 1)
             else "GPU INT8")
    print(f"[측정] 가속기    : {accel}")
    print(f"[측정] 비디오    : {args.video}")
    print(f"[측정] 샘플 주기 : {interval_ms} ms")
    print(f"[측정] 출력 CSV  : {output_csv}\n")

    m  = Measurement()
    t0 = time.time()

    t_tegra = threading.Thread(
        target=tegrastats_thread, args=(m, interval_ms, t0),
        daemon=True, name="tegrastats",
    )
    t_ds = threading.Thread(
        target=deepstream_thread, args=(m, ds_cmd, t0),
        daemon=True, name="deepstream",
    )

    t_tegra.start()
    t_ds.start()

    t_ds.join()
    m.ds_done.set()
    t_tegra.join(timeout=3.0)

    # fps_window가 없는 행에 해당 구간 FPS 소급 적용
    fps_windows = m.get_fps_windows()
    for r in m.rows:
        if r["fps_window"] is not None:
            continue
        for (t_start, t_end, fps) in fps_windows:
            if t_start <= r["elapsed_s"] <= t_end:
                r["fps_window"] = fps
                break

    save_csv(m.rows, output_csv)
    print_summary(m.rows)


if __name__ == "__main__":
    main()
