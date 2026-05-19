#!/usr/bin/env python3
"""
jetson_power_monitor.py
=======================
Jetson INA3221 전력 모니터를 sysfs 에서 직접 읽어 소비전력을 측정.

각 전력 레일(VDD_IN, VDD_CPU_GPU_CV 등)의 전압·전류·전력을
주기적으로 샘플링하여 콘솔에 표시하고 종료 시 통계를 출력.

[측정 원리]
  /sys/class/hwmon/hwmon*/  에서 INA3221 장치를 자동 탐색.
  in{N}_input  (mV) × curr{N}_input (mA) / 1000 = power (mW)

[실행]
  python3 jetson_power_monitor.py
  python3 jetson_power_monitor.py --interval 0.5
  python3 jetson_power_monitor.py --duration 60
  python3 jetson_power_monitor.py --output power_log.csv
"""

import os
import sys
import glob
import time
import csv
import signal
import argparse
import statistics
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 데이터 구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Channel:
    name:      str   # 레일 이름 (예: VDD_IN)
    volt_path: str   # in{N}_input  경로 (mV)
    curr_path: str   # curr{N}_input 경로 (mA)

    def read(self) -> tuple[float, float, float]:
        """(voltage_mV, current_mA, power_mW) 반환."""
        with open(self.volt_path) as f:
            v = float(f.read())
        with open(self.curr_path) as f:
            i = float(f.read())
        return v, i, v * i / 1000.0


@dataclass
class Sensor:
    hwmon_path:  str
    device_name: str
    channels:    list[Channel] = field(default_factory=list)


@dataclass
class Sample:
    timestamp: float
    # {레일 이름: (voltage_mV, current_mA, power_mW)}
    readings:  dict[str, tuple[float, float, float]] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# 센서 자동 탐색
# ══════════════════════════════════════════════════════════════════════════════

def discover_sensors() -> list[Sensor]:
    """
    /sys/class/hwmon/hwmon* 를 스캔하여 INA3221 장치와 채널을 탐색.
    """
    sensors = []
    for hwmon_dir in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        name_file = os.path.join(hwmon_dir, "name")
        if not os.path.exists(name_file):
            continue
        with open(name_file) as f:
            dev_name = f.read().strip()
        if "ina3221" not in dev_name:
            continue

        sensor = Sensor(hwmon_path=hwmon_dir, device_name=dev_name)

        for ch in (1, 2, 3):
            volt_path = os.path.join(hwmon_dir, f"in{ch}_input")
            curr_path = os.path.join(hwmon_dir, f"curr{ch}_input")
            if not (os.path.exists(volt_path) and os.path.exists(curr_path)):
                continue

            label_path = os.path.join(hwmon_dir, f"in{ch}_label")
            if os.path.exists(label_path):
                with open(label_path) as f:
                    rail_name = f.read().strip()
            else:
                rail_name = f"{dev_name}_CH{ch}"

            sensor.channels.append(Channel(
                name=rail_name,
                volt_path=volt_path,
                curr_path=curr_path,
            ))

        if sensor.channels:
            sensors.append(sensor)

    return sensors


# ══════════════════════════════════════════════════════════════════════════════
# 샘플 수집
# ══════════════════════════════════════════════════════════════════════════════

def collect_sample(sensors: list[Sensor]) -> Sample:
    sample = Sample(timestamp=time.time())
    for sensor in sensors:
        for ch in sensor.channels:
            try:
                sample.readings[ch.name] = ch.read()
            except OSError:
                pass
    return sample


# ══════════════════════════════════════════════════════════════════════════════
# 콘솔 출력
# ══════════════════════════════════════════════════════════════════════════════

_COL = 22

def format_table(sample: Sample) -> str:
    lines = [
        f"{'레일':<{_COL}} {'전압(mV)':>10} {'전류(mA)':>10} {'전력(mW)':>10} {'전력(W)':>9}",
        "─" * (_COL + 44),
    ]
    total_mw = 0.0
    for name, (v, i, p) in sample.readings.items():
        lines.append(
            f"{name:<{_COL}} {v:>10.1f} {i:>10.1f} {p:>10.1f} {p/1000:>9.3f}"
        )
        total_mw += p
    lines.append("─" * (_COL + 44))
    lines.append(
        f"{'전체 합계':<{_COL}} {'':>10} {'':>10} {total_mw:>10.1f} {total_mw/1000:>9.3f}"
    )
    return "\n".join(lines)


def print_summary(samples: list[Sample], elapsed: float) -> None:
    if not samples:
        print("수집된 샘플 없음")
        return

    all_names = list(samples[0].readings.keys())

    print("\n" + "═" * 68)
    print("  최종 통계")
    print("═" * 68)
    print(f"{'레일':<{_COL}} {'최소(mW)':>10} {'평균(mW)':>10} {'최대(mW)':>10} {'평균(W)':>9}")
    print("─" * 68)

    total_min = total_avg = total_max = 0.0
    for name in all_names:
        powers = [s.readings[name][2] for s in samples if name in s.readings]
        if not powers:
            continue
        mn  = min(powers)
        avg = statistics.mean(powers)
        mx  = max(powers)
        total_min += mn
        total_avg += avg
        total_max += mx
        print(f"{name:<{_COL}} {mn:>10.1f} {avg:>10.1f} {mx:>10.1f} {avg/1000:>9.3f}")

    print("─" * 68)
    print(f"{'전체 합계':<{_COL}} {total_min:>10.1f} {total_avg:>10.1f} {total_max:>10.1f} {total_avg/1000:>9.3f}")

    # VDD_IN 또는 첫 번째 레일을 에너지 계산 기준으로 표시
    primary = next(
        (n for n in all_names if "vdd_in" in n.lower() or "total" in n.lower()),
        all_names[0] if all_names else None,
    )
    if primary:
        powers     = [s.readings[primary][2] for s in samples if primary in s.readings]
        avg_w      = statistics.mean(powers) / 1000.0
        energy_mwh = avg_w * 1000.0 * elapsed / 3600.0

        print(f"\n기준 레일  : {primary}")
        print(f"평균 전력  : {avg_w * 1000:.1f} mW  ({avg_w:.3f} W)")
        print(f"소비 에너지: {energy_mwh:.4f} mWh  ({elapsed:.0f}초 기준)")

    print(f"\n측정 시간  : {elapsed:.1f}초")
    print(f"샘플 수    : {len(samples)}개  (실제 간격 "
          f"{elapsed / max(len(samples) - 1, 1):.2f}초/샘플)")
    print("═" * 68)


# ══════════════════════════════════════════════════════════════════════════════
# CSV 저장
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(samples: list[Sample], output_path: str) -> None:
    if not samples:
        return
    all_names = list(samples[0].readings.keys())
    headers = ["timestamp", "elapsed_s"]
    for name in all_names:
        headers += [f"{name}_mV", f"{name}_mA", f"{name}_mW"]
    headers.append("total_mW")

    t0 = samples[0].timestamp
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in samples:
            row = [f"{s.timestamp:.3f}", f"{s.timestamp - t0:.3f}"]
            total_mw = 0.0
            for name in all_names:
                if name in s.readings:
                    v, i, p = s.readings[name]
                    row += [f"{v:.1f}", f"{i:.1f}", f"{p:.1f}"]
                    total_mw += p
                else:
                    row += ["", "", ""]
            row.append(f"{total_mw:.1f}")
            writer.writerow(row)

    print(f"\nCSV 저장: {output_path}  ({len(samples)}샘플, {len(all_names)}레일 + total_mW)")


# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jetson INA3221 소비전력 측정기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "실행 예시:\n"
            "  python3 jetson_power_monitor.py\n"
            "  python3 jetson_power_monitor.py --interval 0.5 --duration 60\n"
            "  python3 jetson_power_monitor.py --output power_log.csv\n"
        ),
    )
    parser.add_argument("--interval", type=float, default=1.0, metavar="초",
                        help="샘플링 간격 초 (기본값: 1.0, 최소: 0.1)")
    parser.add_argument("--duration", type=float, default=None, metavar="초",
                        help="측정 시간 초 (기본값: Ctrl+C 까지)")
    parser.add_argument("--output", default=None, metavar="파일.csv",
                        help="CSV 로그 저장 경로 (기본값: 저장 안 함)")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    interval = max(0.1, args.interval)

    # 센서 탐색
    sensors = discover_sensors()
    if not sensors:
        print("오류: INA3221 전력 모니터를 찾을 수 없음")
        print("  확인: ls /sys/class/hwmon/")
        print("  확인: cat /sys/class/hwmon/hwmon*/name")
        sys.exit(1)

    print("발견된 전력 모니터:")
    for s in sensors:
        print(f"  [{s.hwmon_path}]  {s.device_name}")
        for ch in s.channels:
            print(f"    └─ {ch.name}")
    print()

    samples: list[Sample] = []
    running = True

    def on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    t_start = time.perf_counter()
    prev_lines = 0

    print(f"샘플링 간격: {interval}초"
          + (f"  측정 시간: {args.duration}초" if args.duration else "  (Ctrl+C 로 종료)")
          + (f"  → {args.output}" if args.output else ""))
    print()

    try:
        while running:
            elapsed = time.perf_counter() - t_start
            if args.duration and elapsed >= args.duration:
                break

            sample = collect_sample(sensors)
            samples.append(sample)

            # 이전 출력 덮어쓰기 (ANSI 커서 이동)
            if prev_lines:
                sys.stdout.write(f"\033[{prev_lines}A\033[J")

            ts_str = time.strftime("%H:%M:%S", time.localtime(sample.timestamp))
            header = f"[{ts_str}]  경과: {elapsed:7.1f}초  샘플: {len(samples):5d}개"
            print(header)

            table = format_table(sample)
            print(table)

            prev_lines = header.count("\n") + 1 + table.count("\n") + 1
            sys.stdout.flush()

            time.sleep(interval)

    finally:
        elapsed = time.perf_counter() - t_start
        print_summary(samples, elapsed)
        if args.output:
            save_csv(samples, args.output)


if __name__ == "__main__":
    main()
