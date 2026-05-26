#!/usr/bin/env python3
"""
deepstream_yolov8_video.py
==============================
비디오 파일 YOLOv8m 추론 (DeepStream, DLA INT8 기본)

[구조]
  파이프라인 : filesrc → qtdemux → h264parse → nvv4l2decoder
                      → nvvideoconvert → capsfilter(NVMM,NV12)
                      → nvstreammux(batch=1) → nvinfer(DLA INT8) → probe → fakesink

  * --dla-core 0  → DLA Core 0 INT8 (기본)
  * --dla-core 1  → DLA Core 1 INT8
  * --dla-core 2  → GPU INT8 (0·1 이외의 값)

실행:
  python3 deepstream_yolov8_video.py                              # DLA Core 0 INT8 (기본)
  python3 deepstream_yolov8_video.py --dla-core 1                 # DLA Core 1 INT8
  python3 deepstream_yolov8_video.py --dla-core 2                 # GPU INT8
  python3 deepstream_yolov8_video.py --video /path/to/video.mp4
  python3 deepstream_yolov8_video.py --save-json results.json     # 추론 결과 JSON 저장
  python3 deepstream_yolov8_video.py --save-csv  results.csv      # 추론 결과 CSV 저장
  python3 deepstream_yolov8_video.py --power-log power.csv        # 소비 전력 CSV 저장
  python3 deepstream_yolov8_video.py --no-display                 # 화면 표시 비활성화
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
gi.require_version("GstPbutils", "1.0")
from gi.repository import Gst, GLib, GstPbutils
import pyds

# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



BASE_DIR    = "/home/nvidia/workspace/deepstream_yolo"
PGIE_CONFIG = f"{BASE_DIR}/config_infer_yolov8_gpu_int8.txt"
DLA_CONFIGS = {
    0: f"{BASE_DIR}/config_infer_yolov8_dla0_int8.txt",
    1: f"{BASE_DIR}/config_infer_yolov8_dla1_int8.txt",
}
LABEL_FILE      = f"{BASE_DIR}/coco_labels.txt"
DEFAULT_VIDEO   = f"{BASE_DIR}/video_2.mp4"

FPS_LOG_INTERVAL = 5.0


def _find_gpu_util_path() -> str | None:
    """Jetson GPU 사용률 sysfs 경로 탐색. 원시값 범위: 0-1000 (permille)."""
    import glob as _glob
    candidates = [
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/gpu.0/load",
        *_glob.glob("/sys/devices/platform/bus@0/*.gpu/load"),
        *_glob.glob("/sys/devices/platform/*.gpu/load"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _read_gpu_util(path: str) -> float:
    """sysfs 반환값 0-1000 (permille) → 0.0-100.0 (%)로 변환."""
    with open(path) as f:
        return float(f.read().strip()) / 10.0


def _discover_video_size(video_path: str) -> tuple[int, int]:
    """GstPbutils.Discoverer로 파이프라인 실행 전에 입력 해상도를 조회."""
    uri = "file://" + os.path.abspath(video_path)
    discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
    info = discoverer.discover_uri(uri)
    for stream in info.get_video_streams():
        return stream.get_width(), stream.get_height()
    raise RuntimeError(f"비디오 스트림 정보를 찾을 수 없음: {video_path}")


def _load_labels(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f]

LABELS: list = _load_labels(LABEL_FILE)


def _make(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"GStreamer 플러그인 없음: {factory}")
    return el


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


def _save_results_csv(results: list, output_path: str) -> None:
    """추론 결과를 CSV로 저장. 탐지 1건 = 1행, 탐지 없는 프레임은 빈 행 1개."""
    headers = ["frame", "num_detections", "frame_interval_ms", "fps", "gpu_util_pct",
               "class_id", "label", "confidence",
               "bbox_left", "bbox_top", "bbox_width", "bbox_height"]
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in results:
            frame    = r["frame"]
            dets     = r["detections"]
            n        = len(dets)
            interval = r.get("frame_interval_ms")
            fps_val  = f"{1000.0 / interval:.2f}" if interval else ""
            gpu_util = r.get("gpu_util_pct", "")
            gpu_str  = f"{gpu_util:.1f}" if isinstance(gpu_util, float) else ""
            if dets:
                for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
                    l, t, bw, bh = d["bbox_ltwh"]
                    w.writerow([frame, n, interval if interval is not None else "", fps_val,
                                gpu_str,
                                d["class_id"], d["label"], f"{d['confidence']:.4f}",
                                f"{l:.1f}", f"{t:.1f}", f"{bw:.1f}", f"{bh:.1f}"])
            else:
                w.writerow([frame, 0, interval if interval is not None else "", fps_val,
                            gpu_str, "", "", "", "", "", "", ""])
    total_dets = sum(len(r["detections"]) for r in results)
    logger.info("추론 결과 CSV 저장: %s  (%d프레임, 탐지 %d개, 평균 %.1f개/프레임)",
                output_path, len(results), total_dets,
                total_dets / max(len(results), 1))


def _save_json_results(results: list, output_path: str, meta: dict) -> None:
    with open(output_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    total_dets = sum(len(r["detections"]) for r in results)
    logger.info("추론 결과 저장: %s  (%d프레임, 탐지 %d개, 평균 %.1f개/프레임)",
                output_path, len(results), total_dets,
                total_dets / max(len(results), 1))


# ══════════════════════════════════════════════════════════════════════════════
class VideoPipeline:
    """
    filesrc 기반 비디오 파일 추론 파이프라인.
    filesrc → qtdemux → h264parse → nvv4l2decoder → nvvideoconvert
            → capsfilter(NVMM,NV12) → nvstreammux → nvinfer → probe → fakesink
    """

    def __init__(self, video_path: str, config: str,
                 display: bool = False, save_json: str | None = None,
                 save_csv: str | None = None, power_log: str | None = None,
                 conf_threshold: float = 0.25):
        self._video_path     = video_path
        self._config         = config
        self._display        = display
        self._save_json      = save_json
        self._save_csv       = save_csv
        self._power_log      = power_log
        self._conf_threshold = conf_threshold
        self._loop           = GLib.MainLoop()

        self._proc_cnt           = 0
        self._frame_cnt          = 0
        self._t_start            = 0.0
        self._t_fps_log          = 0.0
        self._elapsed            = 0.0
        self._avg_fps            = 0.0
        self._first_frame_t      = 0.0   # 첫 프레임 probe 도달 시각 (워밍업 제외 기준점)
        self._last_probe_t       = 0.0   # 직전 프레임 probe 시각
        self._fps_window_frames  = 0     # 직전 FPS 로그 시점의 프레임 수
        self._results        : list = []
        self._power_samples  : list = []
        self._power_active   : bool = False
        self._gpu_util_pct   : float = 0.0   # 최근 GPU 사용률 (샘플링 스레드가 갱신)
        self._gpu_util_path  : str | None = _find_gpu_util_path()

        self._mux_w, self._mux_h = _discover_video_size(video_path)
        logger.info("입력 해상도: %d×%d", self._mux_w, self._mux_h)

        self._pipeline = self._build()

    # ── 파이프라인 조립 ─────────────────────────────────────────────────────
    def _build(self) -> Gst.Pipeline:
        pipeline = Gst.Pipeline()

        src    = _make("filesrc",        "src")
        demux  = _make("qtdemux",        "demux")
        parse  = _make("h264parse",      "parse")
        dec    = _make("nvv4l2decoder",  "dec")
        nvconv = _make("nvvideoconvert", "nvconv")

        capsfil = _make("capsfilter", "capsfil")
        capsfil.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12"
        ))

        src.set_property("location", self._video_path)

        mux = _make("nvstreammux", "mux")
        mux.set_property("width",                self._mux_w)
        mux.set_property("height",               self._mux_h)
        mux.set_property("batch-size",           1)
        mux.set_property("batched-push-timeout", 100_000)
        mux.set_property("nvbuf-memory-type",    4)
        mux.set_property("live-source",          0)

        inf = _make("nvinfer", "inf")
        inf.set_property("config-file-path", self._config)

        if self._display:
            nvconv2 = _make("nvvideoconvert", "nvconv2")
            osd = _make("nvdsosd", "osd")
            osd.set_property("process-mode", 1)
            osd.set_property("display-text", 1)
            snk = _make("nv3dsink", "snk")
            snk.set_property("sync", False)
            post_inf = (nvconv2, osd, snk)
        else:
            snk = _make("fakesink", "snk")
            snk.set_property("sync", False)
            post_inf = (snk,)

        for el in (src, demux, parse, dec, nvconv, capsfil, mux, inf, *post_inf):
            pipeline.add(el)

        if not src.link(demux):
            raise RuntimeError("filesrc → qtdemux 링크 실패")
        demux.connect("pad-added", self._on_demux_pad_added, parse)

        decode_chain = (parse, dec, nvconv, capsfil)
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
        nvconv.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._resolution_probe
        )

        return pipeline

    # ── 해상도 비교 (첫 번째 버퍼에서 한 번만 실행) ────────────────────────
    def _resolution_probe(self, pad, info) -> Gst.PadProbeReturn:
        caps = pad.get_current_caps()
        if caps:
            s = caps.get_structure(0)
            ok_w, w = s.get_int("width")
            ok_h, h = s.get_int("height")
            if ok_w and ok_h:
                logger.info(
                    "해상도 확인  디코더: %d×%d  /  nvstreammux: %d×%d",
                    w, h, self._mux_w, self._mux_h,
                )
        return Gst.PadProbeReturn.REMOVE  # 첫 프레임 이후 자동 제거

    # ── qtdemux 동적 패드 연결 ──────────────────────────────────────────────
    def _on_demux_pad_added(self, element: Gst.Element, pad: Gst.Pad,
                            h264parse: Gst.Element) -> None:
        """qtdemux가 비디오 스트림 패드를 생성할 때 h264parse sink에 연결."""
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps and "video" in caps.to_string():
            sink_pad = h264parse.get_static_pad("sink")
            if not sink_pad.is_linked():
                ret = pad.link(sink_pad)
                if ret != Gst.PadLinkReturn.OK:
                    logger.error("qtdemux → h264parse 패드 링크 실패: %s", ret)
                else:
                    logger.info("비디오 스트림 연결됨: %s", caps.to_string())

    # ── 전력 샘플링 스레드 ─────────────────────────────────────────────────
    def _power_loop(self, sensors: list) -> None:
        while self._power_active:
            self._power_samples.append(_collect_power_sample(sensors))
            time.sleep(0.5)

    # ── GPU 사용률 샘플링 스레드 ────────────────────────────────────────────
    def _gpu_util_loop(self) -> None:
        while self._power_active:
            try:
                self._gpu_util_pct = _read_gpu_util(self._gpu_util_path)
            except OSError:
                pass
            time.sleep(0.1)

    # ── 탐지 결과 probe ─────────────────────────────────────────────────────
    def _detection_probe(self, pad, info) -> Gst.PadProbeReturn:
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        batch_probe_t = time.perf_counter()

        # 첫 프레임 도달 시각 기록 (워밍업 기준점)
        if self._first_frame_t == 0.0:
            self._first_frame_t = batch_probe_t

        # 프레임 간격: 직전 probe 시각이 있을 때만 계산 (첫 프레임 제외)
        if self._last_probe_t > 0.0:
            frame_interval_ms = round((batch_probe_t - self._last_probe_t) * 1000.0, 2)
        else:
            frame_interval_ms = None

        fl = batch_meta.frame_meta_list
        while fl:
            try:
                fm = pyds.NvDsFrameMeta.cast(fl.data)
            except StopIteration:
                break

            frame_num = self._frame_cnt
            self._frame_cnt += 1

            dets = []
            ol = fm.obj_meta_list
            while ol:
                try:
                    om    = pyds.NvDsObjectMeta.cast(ol.data)
                    r     = om.rect_params
                    label = (LABELS[om.class_id]
                             if 0 <= om.class_id < len(LABELS)
                             else str(om.class_id))
                    conf = float(om.confidence)
                    if conf < self._conf_threshold:
                        ol = ol.next
                        continue
                    dets.append({
                        "class_id":   int(om.class_id),
                        "label":      label,
                        "confidence": round(conf, 4),
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

            if self._save_json or self._save_csv:
                self._results.append({
                    "frame":            frame_num,
                    "frame_interval_ms": frame_interval_ms,
                    "gpu_util_pct":     self._gpu_util_pct,
                    "detections":       dets,
                })

            self._log_fps()

            try:
                fl = fl.next
            except StopIteration:
                break

        self._last_probe_t = batch_probe_t
        return Gst.PadProbeReturn.OK

    def _log_fps(self) -> None:
        now = time.perf_counter()
        if now - self._t_fps_log >= FPS_LOG_INTERVAL:
            window_frames  = self._proc_cnt - self._fps_window_frames
            window_elapsed = now - self._t_fps_log
            window_fps     = window_frames / window_elapsed if window_elapsed > 0 else 0
            logger.info("처리량: %d프레임  구간 %.2f FPS  (%.1fs / %d프레임)",
                        self._proc_cnt, window_fps, window_elapsed, window_frames)
            self._fps_window_frames = self._proc_cnt
            self._t_fps_log = now

    # ── 버스 콜백 ───────────────────────────────────────────────────────────
    def _bus_call(self, bus, msg) -> bool:
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._elapsed = time.perf_counter() - self._t_start
            # 첫 프레임~마지막 프레임 간격 기반 FPS (워밍업 제외)
            frame_span = self._last_probe_t - self._first_frame_t
            if frame_span > 0 and self._proc_cnt > 1:
                self._avg_fps = (self._proc_cnt - 1) / frame_span
            else:
                self._avg_fps = 0.0
            logger.info("EOS — %d프레임 / 평균 %.2f FPS  (프레임 간격 기준, 워밍업 제외)",
                        self._proc_cnt, self._avg_fps)
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

        _power_thread    = None
        _gpu_util_thread = None
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

        if self._save_csv and self._gpu_util_path:
            self._power_active = True
            _gpu_util_thread = threading.Thread(
                target=self._gpu_util_loop, daemon=True, name="gpu-util"
            )
            _gpu_util_thread.start()
            logger.info("GPU 사용률 모니터링 시작: %s", self._gpu_util_path)
        elif self._save_csv:
            logger.warning("GPU 사용률 sysfs 경로 미발견 — gpu_util_pct 미측정")

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("파이프라인 PLAYING 전환 실패")

        self._t_start = self._t_fps_log = time.perf_counter()

        try:
            self._loop.run()
        except KeyboardInterrupt:
            logger.info("사용자 중단 (Ctrl+C)")
        finally:
            self._power_active = False
            if _power_thread and _power_thread.is_alive():
                _power_thread.join(timeout=2.0)
            if _gpu_util_thread and _gpu_util_thread.is_alive():
                _gpu_util_thread.join(timeout=1.0)
            self._pipeline.set_state(Gst.State.NULL)

        # ── 결과 저장 ──────────────────────────────────────────────────────
        if self._save_json and self._results:
            _save_json_results(self._results, self._save_json, {
                "video":       self._video_path,
                "config":      self._config,
                "total_frames": self._proc_cnt,
                "elapsed_s":   round(self._elapsed, 3),
                "avg_fps":     round(self._avg_fps, 2),
                "mux_width":   self._mux_w,
                "mux_height":  self._mux_h,
                "bbox_format": f"left, top, width, height (mux {self._mux_w}×{self._mux_h} 좌표계)",
            })

        if self._save_csv and self._results:
            _save_results_csv(self._results, self._save_csv)

        if self._power_log and self._power_samples:
            _save_power_csv(self._power_samples, self._power_log, self._elapsed)


# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="비디오 파일 YOLOv8m 추론 (기본: DLA Core 0 INT8)"
    )
    p.add_argument("--video",      default=DEFAULT_VIDEO, metavar="PATH",
                   help=f"입력 비디오 파일 (기본: {DEFAULT_VIDEO})")
    p.add_argument("--config",     default=None,
                   help="nvinfer 설정 파일 직접 지정 (옵션)")
    p.add_argument("--dla-core",   type=int, default=0,
                   help="가속기 선택: 0=DLA Core 0 (기본), 1=DLA Core 1, 그 외=GPU INT8")
    p.add_argument("--no-display", action="store_true",
                   help="화면 표시 비활성화 (기본: 표시 ON)")
    p.add_argument("--save-json",  default=None, metavar="PATH",
                   help="추론 결과를 JSON으로 저장 (미지정 시 저장 안 함)")
    p.add_argument("--save-csv",   default=None, metavar="PATH",
                   help="추론 결과를 CSV로 저장, 탐지 1건=1행 (미지정 시 저장 안 함)")
    p.add_argument("--power-log",  default=None, metavar="PATH",
                   help="추론 중 소비 전력을 CSV로 저장 (미지정 시 측정 안 함)")
    p.add_argument("--conf-threshold", type=float, default=0.6, metavar="CONF",
                   help="출력 confidence 임계값 (기본: 0.25)")
    p.add_argument("--debug",      action="store_true", help="GStreamer 디버그 로그")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        os.environ.setdefault("GST_DEBUG", "3")

    Gst.init(None)

    if args.dla_core in (0, 1):
        config = args.config or DLA_CONFIGS[args.dla_core]
        logger.info("DLA Core %d INT8", args.dla_core)
    else:
        config = args.config or PGIE_CONFIG
        logger.info("GPU INT8")

    if not os.path.isfile(config):
        logger.error("설정 파일 없음: %s", config)
        sys.exit(1)

    if not os.path.isfile(args.video):
        logger.error("비디오 파일 없음: %s", args.video)
        sys.exit(1)

    display = not args.no_display
    if display and not os.environ.get("DISPLAY"):
        logger.error("디스플레이 출력에 DISPLAY 환경변수가 필요합니다. (예: DISPLAY=:0)\n"
                     "비활성화하려면 --no-display 옵션을 사용하세요.")
        sys.exit(1)

    logger.info("입력 비디오: %s", args.video)
    logger.info("설정: %s", config)
    logger.info("confidence 임계값: %.2f", args.conf_threshold)

    if display:
        logger.info("디스플레이: ON (nvdsosd + nv3dsink, sync=False)")
    if args.save_json:
        logger.info("추론 결과 저장 (JSON): %s", args.save_json)
    if args.save_csv:
        logger.info("추론 결과 저장 (CSV): %s", args.save_csv)
    if args.power_log:
        logger.info("전력 로그 저장: %s", args.power_log)

    VideoPipeline(
        video_path=args.video,
        config=config,
        display=display,
        save_json=args.save_json,
        save_csv=args.save_csv,
        power_log=args.power_log,
        conf_threshold=args.conf_threshold,
    ).run()


if __name__ == "__main__":
    main()
