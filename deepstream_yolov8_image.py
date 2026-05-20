#!/usr/bin/env python3
"""
deepstream_yolov8_image.py
==============================
단일 채널 이미지 스트림 YOLOv8m 추론 (DeepStream, DLA INT8 기본)

[구조]
  파이프라인 : appsrc(JPEG bytes 1장 = 버퍼 1개)
                      → jpegparse → jpegdec → nvvideoconvert
                      → capsfilter(NVMM,NV12) → nvstreammux(batch=1)
                      → nvinfer(DLA INT8) → probe → fakesink

  * JPEG 파일 1개를 Gst.Buffer 1개로 밀어넣어 명확한 frame 경계 보장
  * --dla-core 0  → DLA Core 0 INT8 (기본)
  * --dla-core 1  → DLA Core 1 INT8
  * --dla-core 2  → GPU FP16 (0·1 이외의 값)

실행:
  python3 deepstream_yolov8_image.py                              # DLA Core 0 INT8 (기본)
  python3 deepstream_yolov8_image.py --dla-core 1                 # DLA Core 1 INT8
  python3 deepstream_yolov8_image.py --dla-core 2                 # GPU FP16
  python3 deepstream_yolov8_image.py --image-dir /path/to/images
  python3 deepstream_yolov8_image.py --output-dir /path/to/output # 바운딩박스 이미지 저장
  python3 deepstream_yolov8_image.py --save-json results.json     # 추론 결과 JSON 저장
  python3 deepstream_yolov8_image.py --power-log power.csv        # 소비 전력 CSV 저장
"""

import sys
import os
import time
import argparse
import logging
import threading
import json
import csv
import statistics
from dataclasses import dataclass, field as dc_field

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

from PIL import Image, ImageDraw

# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _redirect_c_fds_to_devnull() -> None:
    """
    C 라이브러리(NvMMLite/VIC, GStreamer)가 fd 1/fd 2 에 직접 쓰는 노이즈 로그를
    /dev/null 로 우회한다.

    원리:
      saved_fd1 = dup(1),  saved_fd2 = dup(2)   ← 원래 터미널 fd 복사
      dup2(/dev/null, 1),  dup2(/dev/null, 2)    ← C 코드 출력 버림
      sys.stdout = fdopen(saved_fd1)             ← Python print/logging → 터미널
      sys.stderr = fdopen(saved_fd2)

    결과: NvMMLiteOpen, GStreamer INFO 노이즈 억제
          Python logging(INFO/ERROR) 및 탐지 결과는 터미널에 그대로 표시
    """
    sys.stdout.flush()
    sys.stderr.flush()

    saved_fd1 = os.dup(1)
    saved_fd2 = os.dup(2)
    devnull   = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    sys.stdout = os.fdopen(saved_fd1, "w", buffering=1)
    sys.stderr = os.fdopen(saved_fd2, "w", buffering=1)

    # logging 핸들러를 새 sys.stdout 으로 교체
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout

BASE_DIR        = "/home/nvidia/workspace/deepstream_yolo"
PGIE_CONFIG     = f"{BASE_DIR}/config_infer_yolov8_gpu_fp16.txt"
DLA_CONFIGS     = {
    0: f"{BASE_DIR}/config_infer_yolov8_dla0_int8.txt",
    1: f"{BASE_DIR}/config_infer_yolov8_dla1_int8.txt",
}
LABEL_FILE      = f"{BASE_DIR}/coco_labels.txt"
DEFAULT_IMG_DIR = "/home/nvidia/workspace/val2017"

MUX_W = 640
MUX_H = 640

FPS_LOG_INTERVAL = 5.0

FRAME_RATE_N = 30
FRAME_RATE_D = 1


def _load_labels(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f]

LABELS: list = _load_labels(LABEL_FILE)


def _make(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"GStreamer 플러그인 없음: {factory}")
    return el


# 80 COCO 클래스용 색상 팔레트 (class_id % 20)
_BBOX_COLORS = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]

def _bbox_color(class_id: int) -> tuple:
    return _BBOX_COLORS[class_id % len(_BBOX_COLORS)]


def _draw_and_save(img_path: str, dets: list, output_dir: str) -> None:
    """탐지 결과(바운딩박스 + 라벨)를 원본 이미지에 그려 output_dir 에 저장."""
    img  = Image.open(img_path).convert("RGB")
    iw, ih = img.size
    # rect_params 좌표는 nvstreammux 출력(MUX_W×MUX_H) 공간 기준이므로
    # 원본 이미지 해상도로 역변환이 필요하다.
    sx, sy = iw / MUX_W, ih / MUX_H
    draw = ImageDraw.Draw(img)
    for det in dets:
        l, t, w, h = det["bbox_ltwh"]
        x1 = int(l * sx)
        y1 = int(t * sy)
        x2 = int((l + w) * sx)
        y2 = int((t + h) * sy)
        color = _bbox_color(det["class_id"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{det['label']} {det['confidence']:.2f}"
        tw   = len(text) * 7
        ty   = max(y1 - 17, 0)
        draw.rectangle([x1, ty, x1 + tw, ty + 16], fill=color)
        draw.text((x1 + 2, ty + 1), text, fill=(255, 255, 255))
    img.save(os.path.join(output_dir, os.path.basename(img_path)), "JPEG", quality=92)


# ── 전력 모니터링 (INA3221) ─────────────────────────────────────────────────
@dataclass
class _PwrChannel:
    name: str
    volt_path: str
    curr_path: str

    def read(self) -> tuple[float, float, float]:
        with open(self.volt_path) as f:
            v = float(f.read())
        with open(self.curr_path) as f:
            i = float(f.read())
        return v, i, v * i / 1000.0


@dataclass
class _PwrSensor:
    hwmon_path:  str
    device_name: str
    channels:    list = dc_field(default_factory=list)


@dataclass
class _PwrSample:
    timestamp: float
    readings:  dict = dc_field(default_factory=dict)  # name → (mV, mA, mW)


def _discover_power_sensors() -> list:
    import glob as _glob
    sensors = []
    for hwmon_dir in sorted(_glob.glob("/sys/class/hwmon/hwmon*")):
        name_file = os.path.join(hwmon_dir, "name")
        if not os.path.exists(name_file):
            continue
        with open(name_file) as f:
            dev_name = f.read().strip()
        if "ina3221" not in dev_name:
            continue
        sensor = _PwrSensor(hwmon_path=hwmon_dir, device_name=dev_name)
        for ch in (1, 2, 3):
            vp = os.path.join(hwmon_dir, f"in{ch}_input")
            cp = os.path.join(hwmon_dir, f"curr{ch}_input")
            if not (os.path.exists(vp) and os.path.exists(cp)):
                continue
            lp = os.path.join(hwmon_dir, f"in{ch}_label")
            rail = open(lp).read().strip() if os.path.exists(lp) else f"{dev_name}_CH{ch}"
            sensor.channels.append(_PwrChannel(name=rail, volt_path=vp, curr_path=cp))
        if sensor.channels:
            sensors.append(sensor)
    return sensors


def _collect_power_sample(sensors: list) -> _PwrSample:
    s = _PwrSample(timestamp=time.time())
    for sensor in sensors:
        for ch in sensor.channels:
            try:
                s.readings[ch.name] = ch.read()
            except OSError:
                pass
    return s


def _save_power_csv(samples: list, output_path: str, elapsed: float) -> None:
    if not samples:
        return
    all_names = list(samples[0].readings.keys())
    headers   = ["timestamp", "elapsed_s"]
    for n in all_names:
        headers += [f"{n}_mV", f"{n}_mA", f"{n}_mW"]
    headers.append("total_mW")

    t0 = samples[0].timestamp
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for s in samples:
            row   = [f"{s.timestamp:.3f}", f"{s.timestamp - t0:.3f}"]
            total = 0.0
            for n in all_names:
                if n in s.readings:
                    v, i, p = s.readings[n]
                    row += [f"{v:.1f}", f"{i:.1f}", f"{p:.1f}"]
                    total += p
                else:
                    row += ["", "", ""]
            row.append(f"{total:.1f}")
            w.writerow(row)

        # 통계 요약
        w.writerow([])
        w.writerow(["# 통계", "", "min_mW", "avg_mW", "max_mW"])
        total_avg = 0.0
        for n in all_names:
            powers = [s.readings[n][2] for s in samples if n in s.readings]
            if not powers:
                continue
            avg = statistics.mean(powers)
            total_avg += avg
            w.writerow([n, "", f"{min(powers):.1f}", f"{avg:.1f}", f"{max(powers):.1f}"])
        w.writerow(["total", "", "", f"{total_avg:.1f}", ""])
        w.writerow(["elapsed_s", "", f"{elapsed:.1f}"])
        w.writerow(["samples",   "", f"{len(samples)}"])

    logger.info("전력 로그 저장: %s  (%d샘플, %.1f초)", output_path, len(samples), elapsed)
    logger.info("  평균 총 전력: %.1f mW  (%.3f W)", total_avg, total_avg / 1000.0)


def _save_json_results(results: list, output_path: str, meta: dict) -> None:
    with open(output_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    total_dets = sum(len(r["detections"]) for r in results)
    logger.info("추론 결과 저장: %s  (%d장, 탐지 %d개, 평균 %.1f개/장)",
                output_path, len(results), total_dets,
                total_dets / max(len(results), 1))


def _collect_images(img_dir: str, max_images: int | None = None) -> list:
    """디렉토리에서 JPEG 파일 절대 경로 목록을 정렬하여 반환."""
    files = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith('.jpg')
    )
    if not files:
        raise ValueError(f"JPEG 파일 없음: {img_dir}")
    if max_images is not None:
        files = files[:max_images]
    return files


# ══════════════════════════════════════════════════════════════════════════════
class StreamPipeline:
    """
    appsrc 기반 단일 파이프라인 이미지 스트림 추론.
    JPEG 파일 1개를 Gst.Buffer 1개 단위로 밀어넣어 명확한 frame 경계를 보장.
    push 전용 스레드가 GStreamer 스트리밍 스레드와 독립적으로 동작한다.
    """

    def __init__(self, img_dir: str, config: str,
                 max_images: int | None = None, output_dir: str | None = None,
                 display: bool = False, save_json: str | None = None,
                 power_log: str | None = None):
        self._config     = config
        self._output_dir = output_dir
        self._display    = display
        self._save_json  = save_json
        self._power_log  = power_log
        self._loop       = GLib.MainLoop()

        self._proc_cnt   = 0
        self._t_start    = 0.0
        self._t_fps_log  = 0.0
        self._elapsed    = 0.0
        self._avg_fps    = 0.0
        self._probe_idx  = 0
        self._push_thread: threading.Thread | None = None

        self._results       : list = []   # JSON 저장용
        self._power_samples : list = []   # 전력 샘플
        self._power_active  : bool = False

        self._img_files = _collect_images(img_dir, max_images)
        self._total     = len(self._img_files)

        logger.info("이미지 목록: %d장  첫번째: %s  마지막: %s",
                    self._total,
                    os.path.basename(self._img_files[0]),
                    os.path.basename(self._img_files[-1]))

        self._pipeline, self._appsrc = self._build()

    # ── 파이프라인 조립 ─────────────────────────────────────────────────────
    def _build(self) -> tuple:
        pipeline = Gst.Pipeline()

        # appsrc: push 전용 스레드(_push_loop)에서 버퍼를 밀어넣는 소스
        # block=True → 큐가 가득 차면 push 스레드만 블로킹 (스트리밍 스레드 무관)
        src = _make("appsrc", "src")
        src.set_property("caps", Gst.Caps.from_string(
            f"image/jpeg,framerate={FRAME_RATE_N}/{FRAME_RATE_D}"
        ))
        src.set_property("stream-type", 0)          # GST_APP_STREAM_TYPE_STREAM
        src.set_property("format",      Gst.Format.BYTES)
        src.set_property("is-live",     False)
        src.set_property("block",       True)
        src.set_property("max-bytes",   32 * 1024 * 1024)
        # need-data 콜백 없음 — run()에서 별도 스레드가 push를 전담한다

        capsfil = _make("capsfilter", "capsfil")
        capsfil.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12"
        ))

        parse  = _make("jpegparse",      "parse")
        dec    = _make("jpegdec",         "dec")
        nvconv = _make("nvvideoconvert",  "nvconv")
        decode_chain = (src, parse, dec, nvconv, capsfil)

        mux = _make("nvstreammux", "mux")
        mux.set_property("width",                MUX_W)
        mux.set_property("height",               MUX_H)
        mux.set_property("batch-size",           1)
        mux.set_property("batched-push-timeout", 100_000)
        mux.set_property("nvbuf-memory-type",    4)
        mux.set_property("live-source",          0)

        inf = _make("nvinfer", "inf")
        inf.set_property("config-file-path", self._config)

        if self._display:
            nvconv2 = _make("nvvideoconvert", "nvconv2")   # NV12 → RGBA (nvdsosd GPU 모드용)
            osd = _make("nvdsosd", "osd")
            osd.set_property("process-mode", 1)   # GPU 모드
            osd.set_property("display-text", 1)
            snk = _make("nv3dsink", "snk")
            snk.set_property("sync", False)        # DLA GPU 타이머 충돌 방지
            post_inf = (nvconv2, osd, snk)
        else:
            snk = _make("fakesink", "snk")
            snk.set_property("sync", False)
            post_inf = (snk,)

        for el in (*decode_chain, mux, inf, *post_inf):
            pipeline.add(el)

        for a, b in zip(decode_chain, decode_chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"{a.get_name()} → {b.get_name()} 링크 실패")
        capsfil.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        if not mux.link(inf):
            raise RuntimeError("mux → inf 링크 실패")
        prev = inf
        for el in post_inf:
            if not prev.link(el):
                raise RuntimeError(f"{prev.get_name()} → {el.get_name()} 링크 실패")
            prev = el

        inf.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._detection_probe
        )

        return pipeline, src

    # ── push 전용 스레드 ────────────────────────────────────────────────────
    def _push_loop(self) -> None:
        """
        GStreamer 스트리밍 스레드와 독립된 Python 스레드에서 실행.
        JPEG 파일 1개 = Gst.Buffer 1개 단위로 순서대로 push 한다.
        block=True 이므로 appsrc 큐가 가득 차면 이 스레드만 대기하고
        GStreamer 스트리밍 스레드는 계속 동작한다 (deadlock 없음).
        """
        frame_duration = Gst.SECOND * FRAME_RATE_D // FRAME_RATE_N

        for idx, path in enumerate(self._img_files):
            with open(path, "rb") as f:
                data = f.read()

            buf = Gst.Buffer.new_wrapped(data)
            buf.pts      = idx * frame_duration
            buf.dts      = idx * frame_duration
            buf.duration = frame_duration

            ret = self._appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                logger.error("push-buffer 실패: %s  (%s)", ret, os.path.basename(path))
                self._loop.quit()
                return

        self._appsrc.emit("end-of-stream")

    # ── 전력 샘플링 스레드 ─────────────────────────────────────────────────
    def _power_loop(self, sensors: list) -> None:
        while self._power_active:
            self._power_samples.append(_collect_power_sample(sensors))
            time.sleep(0.5)

    # ── 탐지 결과 probe ─────────────────────────────────────────────────────
    def _detection_probe(self, pad, info) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        fl = batch_meta.frame_meta_list
        while fl:
            try:
                fm = pyds.NvDsFrameMeta.cast(fl.data)
            except StopIteration:
                break

            # push 순서 = probe 순서이므로 probe_idx 로 파일명 복원
            idx      = self._probe_idx
            img_name = (os.path.basename(self._img_files[idx])
                        if idx < self._total else f"frame_{idx}")
            self._probe_idx += 1

            dets = []
            ol = fm.obj_meta_list
            while ol:
                try:
                    om    = pyds.NvDsObjectMeta.cast(ol.data)
                    r     = om.rect_params
                    label = (LABELS[om.class_id]
                             if 0 <= om.class_id < len(LABELS)
                             else str(om.class_id))
                    dets.append({
                        "class_id":   int(om.class_id),
                        "label":      label,
                        "confidence": round(float(om.confidence), 4),
                        "bbox_ltwh":  [round(float(r.left),  1), round(float(r.top),    1),
                                       round(float(r.width), 1), round(float(r.height), 1)],
                    })
                    if self._display:
                        om.text_params.display_text = f"{label} {om.confidence:.2f}"
                        om.text_params.x_offset = int(r.left)
                        om.text_params.y_offset = max(0, int(r.top) - 10)
                        om.text_params.font_params.font_name = "Serif"
                        om.text_params.font_params.font_size = 10
                        om.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                        om.text_params.set_bg_clr = 1
                        om.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)
                    ol = ol.next
                except StopIteration:
                    break

            self._proc_cnt += 1
            _print_results(img_name, dets)
            if self._save_json and idx < self._total:
                self._results.append({
                    "image_file": img_name,
                    "detections": dets,
                })
            if self._output_dir and idx < self._total:
                _draw_and_save(self._img_files[idx], dets, self._output_dir)
            self._log_fps()

            try:
                fl = fl.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _log_fps(self) -> None:
        now = time.perf_counter()
        if now - self._t_fps_log >= FPS_LOG_INTERVAL:
            elapsed = now - self._t_start
            logger.info("처리량: %d / %d장  %.1fs  %.2f FPS",
                        self._proc_cnt, self._total, elapsed,
                        self._proc_cnt / elapsed if elapsed > 0 else 0)
            self._t_fps_log = now

    # ── 버스 콜백 ───────────────────────────────────────────────────────────
    def _bus_call(self, bus, msg) -> bool:
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._elapsed = time.perf_counter() - self._t_start
            self._avg_fps = self._proc_cnt / self._elapsed if self._elapsed > 0 else 0
            logger.info("EOS — %d장 / %.1fs / 평균 %.2f FPS",
                        self._proc_cnt, self._elapsed, self._avg_fps)
            self._loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error("파이프라인 오류: %s (%s)", err, dbg)
            self._loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, d = msg.parse_warning()
            logger.warning("%s (%s)", w, d)
        return True

    # ── 실행 ────────────────────────────────────────────────────────────────
    def run(self) -> None:
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

        # 전력 모니터링 시작
        _power_thread = None
        if self._power_log:
            sensors = _discover_power_sensors()
            if sensors:
                self._power_active = True
                _power_thread = threading.Thread(
                    target=self._power_loop, args=(sensors,),
                    daemon=True, name="power-mon"
                )
                _power_thread.start()
                logger.info("전력 모니터링 시작 (0.5s 간격, %d레일)",
                            sum(len(s.channels) for s in sensors))
            else:
                logger.warning("INA3221 전력 모니터 미발견 — power-log 비활성")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("파이프라인 PLAYING 전환 실패")

        self._push_thread = threading.Thread(
            target=self._push_loop, daemon=True, name="jpeg-push"
        )
        self._push_thread.start()

        self._t_start = self._t_fps_log = time.perf_counter()

        try:
            self._loop.run()
        except KeyboardInterrupt:
            logger.info("사용자 중단 (Ctrl+C)")
        finally:
            self._power_active = False
            if _power_thread and _power_thread.is_alive():
                _power_thread.join(timeout=2.0)
            self._pipeline.set_state(Gst.State.NULL)
            if self._push_thread and self._push_thread.is_alive():
                self._push_thread.join(timeout=2.0)

        # ── 결과 저장 ──────────────────────────────────────────────────────
        if self._save_json and self._results:
            _save_json_results(self._results, self._save_json, {
                "config":      self._config,
                "total_images": self._proc_cnt,
                "elapsed_s":   round(self._elapsed, 3),
                "avg_fps":     round(self._avg_fps, 2),
                "mux_width":   MUX_W,
                "mux_height":  MUX_H,
                "bbox_format": "left, top, width, height (mux 640×640 좌표계)",
            })

        if self._power_log and self._power_samples:
            _save_power_csv(self._power_samples, self._power_log, self._elapsed)


# ══════════════════════════════════════════════════════════════════════════════
def _print_results(img_name: str, dets: list) -> None:
    if dets:
        top = max(dets, key=lambda d: d["confidence"])
        l, t, w, h = top["bbox_ltwh"]
        suffix = f"  (+{len(dets)-1})" if len(dets) > 1 else ""
        print(f"{img_name:15s}  탐지={len(dets):3d}  "
              f"{top['label']:15s} {top['confidence']:.3f}"
              f"  [{l:.0f},{t:.0f},{w:.0f}×{h:.0f}]{suffix}")
    else:
        print(f"{img_name:15s}  탐지=  0")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="단일 채널 이미지 스트림 YOLOv8m 추론 (기본: DLA INT8)"
    )
    p.add_argument("--image-dir", default=DEFAULT_IMG_DIR,
                   help=f"이미지 디렉토리 (기본: {DEFAULT_IMG_DIR})")
    p.add_argument("--config",    default=None,
                   help="nvinfer 설정 파일 직접 지정 (옵션)")
    p.add_argument("--dla-core",  type=int, default=0,
                   help="가속기 선택: 0=DLA Core 0 (기본), 1=DLA Core 1, 그 외=GPU FP16")
    p.add_argument("--debug",     action="store_true", help="GStreamer 디버그 로그")
    p.add_argument("--max-images", type=int, default=None,
                   help="테스트용 최대 처리 이미지 수")
    p.add_argument("--output-dir", default=None,
                   help="바운딩박스·라벨을 그린 결과 이미지 저장 디렉토리 (미지정 시 저장 안 함)")
    p.add_argument("--display", action="store_true",
                   help="추론 결과를 실시간 화면에 표시 (DISPLAY=:0 필요, nvdsosd + nv3dsink)")
    p.add_argument("--save-json", default=None, metavar="PATH",
                   help="추론 결과를 JSON으로 저장 (미지정 시 저장 안 함)")
    p.add_argument("--power-log", default=None, metavar="PATH",
                   help="추론 중 소비 전력을 CSV로 저장 (미지정 시 측정 안 함)")
    p.add_argument("--quiet", action="store_true",
                   help="NvMMLite/VIC C 라이브러리 stdout 노이즈 억제")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quiet:
        _redirect_c_fds_to_devnull()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ.setdefault("GST_DEBUG", "3")

    Gst.init(None)

    if args.dla_core in (0, 1):
        config = args.config or DLA_CONFIGS[args.dla_core]
        logger.info("DLA Core %d INT8", args.dla_core)
    else:
        config = args.config or PGIE_CONFIG
        logger.info("GPU FP16")

    if not os.path.isfile(config):
        logger.error("설정 파일 없음: %s", config)
        sys.exit(1)

    if not os.path.isdir(args.image_dir):
        logger.error("이미지 디렉토리 없음: %s", args.image_dir)
        sys.exit(1)

    if args.display and not os.environ.get("DISPLAY"):
        logger.error("--display 옵션은 DISPLAY 환경변수가 필요합니다. (예: DISPLAY=:0)")
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("결과 저장 디렉토리: %s", args.output_dir)

    logger.info("이미지 디렉토리: %s", args.image_dir)
    logger.info("설정: %s", config)
    if args.display:
        logger.info("디스플레이: ON (nvdsosd + nv3dsink, sync=False)")

    if args.save_json:
        logger.info("추론 결과 저장: %s", args.save_json)
    if args.power_log:
        logger.info("전력 로그 저장: %s", args.power_log)

    StreamPipeline(args.image_dir, config,
                   max_images=args.max_images, output_dir=args.output_dir,
                   display=args.display, save_json=args.save_json,
                   power_log=args.power_log).run()


if __name__ == "__main__":
    main()
