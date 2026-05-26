#!/usr/bin/env python3
"""
eval_gpu_map.py
==========================
GPU / DLA INT8 엔진의 COCO val2017 mAP 측정

실행:
  python3 eval_gpu_map.py                    # GPU INT8 (기본)
  python3 eval_gpu_map.py --dla-core 0       # DLA Core 0 INT8
  python3 eval_gpu_map.py --dla-core 1       # DLA Core 1 INT8
  python3 eval_gpu_map.py --max-images 100   # 빠른 테스트
  python3 eval_gpu_map.py --save-dets result_dets.json
  python3 eval_gpu_map.py --quiet
"""

import sys
import os
import time
import argparse
import logging
import threading
import json
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _redirect_c_fds_to_devnull():
    sys.stdout.flush(); sys.stderr.flush()
    saved1 = os.dup(1); saved2 = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2); os.close(dn)
    sys.stdout = os.fdopen(saved1, "w", buffering=1)
    sys.stderr = os.fdopen(saved2, "w", buffering=1)
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout


BASE_DIR    = "/home/nvidia/workspace/deepstream_yolo"
ANN_FILE    = "/home/nvidia/workspace/annotations_trainval2017/annotations/instances_val2017.json"
IMG_DIR     = "/home/nvidia/workspace/val2017"

# 가속기별 mAP 평가용 config (conf=0.001, nms-iou=0.60)
EVAL_CONFIGS = {
    "gpu":  f"{BASE_DIR}/config_infer_yolov8_gpu_int8_eval.txt",
    "dla0": f"{BASE_DIR}/config_infer_yolov8_dla0_int8_eval.txt",
    "dla1": f"{BASE_DIR}/config_infer_yolov8_dla1_int8_eval.txt",
}

MUX_W = MUX_H = 640
FRAME_RATE_N, FRAME_RATE_D = 30, 1


def _build_coco_maps(ann_file: str):
    """COCO 어노테이션에서 필요한 매핑 4종을 반환.

    Returns
    -------
    yolo_to_coco_id : list[int]   길이 80, YOLO class_id(0-79) → COCO category_id
    fname_to_imgid  : dict[str, int]   파일명(확장자 없음) → image_id
    imgid_to_wh     : dict[int, tuple] image_id → (width, height)
    categories      : list[dict]  COCO categories 섹션 그대로 (저장용)
    """
    with open(ann_file) as f:
        data = json.load(f)

    sorted_cats     = sorted(data["categories"], key=lambda c: c["id"])
    yolo_to_coco_id = [c["id"] for c in sorted_cats]
    categories      = [{"id": c["id"], "name": c["name"],
                        "supercategory": c.get("supercategory", "")}
                       for c in sorted_cats]

    fname_to_imgid = {}
    imgid_to_wh    = {}
    for img in data["images"]:
        stem = os.path.splitext(img["file_name"])[0]
        fname_to_imgid[stem] = img["id"]
        imgid_to_wh[img["id"]] = (img["width"], img["height"])

    return yolo_to_coco_id, fname_to_imgid, imgid_to_wh, categories


def _make(factory: str, name: str) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"플러그인 없음: {factory}")
    return el


def _collect_images(img_dir: str, max_images: int | None):
    files = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(".jpg")
    )
    if not files:
        raise ValueError(f"JPEG 없음: {img_dir}")
    return files[:max_images] if max_images else files


# ══════════════════════════════════════════════════════════════════════════════
class EvalPipeline:
    def __init__(self, config: str, img_files: list,
                 yolo_to_coco_id: list, fname_to_imgid: dict, imgid_to_wh: dict):
        self._config          = config
        self._img_files       = img_files
        self._total           = len(img_files)
        self._yolo_to_coco_id = yolo_to_coco_id
        self._fname_to_imgid  = fname_to_imgid
        self._imgid_to_wh     = imgid_to_wh

        self._loop       = GLib.MainLoop()
        self._probe_idx  = 0
        self._proc_cnt   = 0
        self._t_start    = 0.0
        self._push_thread: threading.Thread | None = None

        self.coco_dets: list = []   # COCO 형식 탐지 결과 누적

        self._pipeline, self._appsrc = self._build()

    def _build(self):
        pipeline = Gst.Pipeline()

        src = _make("appsrc", "src")
        src.set_property("caps", Gst.Caps.from_string(
            f"image/jpeg,framerate={FRAME_RATE_N}/{FRAME_RATE_D}"
        ))
        src.set_property("stream-type", 0)
        src.set_property("format",      Gst.Format.BYTES)
        src.set_property("is-live",     False)
        src.set_property("block",       True)
        src.set_property("max-bytes",   32 * 1024 * 1024)

        parse  = _make("jpegparse",     "parse")
        dec    = _make("jpegdec",       "dec")
        nvconv = _make("nvvideoconvert","nvconv")
        capsfil = _make("capsfilter",   "capsfil")
        capsfil.set_property("caps", Gst.Caps.from_string(
            "video/x-raw(memory:NVMM),format=NV12"
        ))
        mux = _make("nvstreammux", "mux")
        mux.set_property("width",                MUX_W)
        mux.set_property("height",               MUX_H)
        mux.set_property("batch-size",           1)
        mux.set_property("batched-push-timeout", 100_000)
        mux.set_property("nvbuf-memory-type",    4)
        mux.set_property("live-source",          0)

        inf = _make("nvinfer", "inf")
        inf.set_property("config-file-path", self._config)

        snk = _make("fakesink", "snk")
        snk.set_property("sync", False)

        for el in (src, parse, dec, nvconv, capsfil, mux, inf, snk):
            pipeline.add(el)

        for a, b in zip((src, parse, dec, nvconv), (parse, dec, nvconv, capsfil)):
            if not a.link(b):
                raise RuntimeError(f"{a.get_name()} → {b.get_name()} 링크 실패")
        capsfil.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        if not mux.link(inf):
            raise RuntimeError("mux → inf 링크 실패")
        if not inf.link(snk):
            raise RuntimeError("inf → snk 링크 실패")

        inf.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._probe
        )
        return pipeline, src

    def _push_loop(self):
        dur = Gst.SECOND * FRAME_RATE_D // FRAME_RATE_N
        for idx, path in enumerate(self._img_files):
            with open(path, "rb") as f:
                data = f.read()
            buf = Gst.Buffer.new_wrapped(data)
            buf.pts = buf.dts = idx * dur
            buf.duration = dur
            ret = self._appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                self._loop.quit()
                return
        self._appsrc.emit("end-of-stream")

    def _probe(self, pad, info) -> Gst.PadProbeReturn:
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

            idx      = self._probe_idx
            self._probe_idx += 1
            self._proc_cnt  += 1

            path     = self._img_files[idx] if idx < self._total else None
            stem     = os.path.splitext(os.path.basename(path))[0] if path else None
            image_id = self._fname_to_imgid.get(stem) if stem else None

            # 원본 이미지 크기 조회 (bbox 역변환용)
            iw, ih = self._imgid_to_wh.get(image_id, (MUX_W, MUX_H)) if image_id else (MUX_W, MUX_H)
            sx = iw / MUX_W
            sy = ih / MUX_H

            ol = fm.obj_meta_list
            while ol:
                try:
                    om = pyds.NvDsObjectMeta.cast(ol.data)
                    r  = om.rect_params

                    # mux(640×640) → 원본 이미지 좌표계
                    x = float(r.left)   * sx
                    y = float(r.top)    * sy
                    w = float(r.width)  * sx
                    h = float(r.height) * sy

                    coco_cat = (self._yolo_to_coco_id[om.class_id]
                                if 0 <= om.class_id < len(self._yolo_to_coco_id)
                                else om.class_id + 1)

                    if image_id is not None:
                        self.coco_dets.append({
                            "image_id":    image_id,
                            "category_id": coco_cat,
                            "bbox":        [round(x, 2), round(y, 2),
                                            round(w, 2), round(h, 2)],
                            "score":       round(float(om.confidence), 4),
                        })
                    ol = ol.next
                except StopIteration:
                    break

            if self._proc_cnt % 500 == 0:
                elapsed = time.perf_counter() - self._t_start
                logger.info("진행 %d/%d  %.1fs  %.2f FPS  탐지 누계 %d",
                            self._proc_cnt, self._total, elapsed,
                            self._proc_cnt / elapsed if elapsed > 0 else 0,
                            len(self.coco_dets))

            try:
                fl = fl.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _bus_call(self, bus, msg) -> bool:
        t = msg.type
        if t == Gst.MessageType.EOS:
            elapsed = time.perf_counter() - self._t_start
            logger.info("EOS — %d장 / %.1fs / %.2f FPS",
                        self._proc_cnt, elapsed,
                        self._proc_cnt / elapsed if elapsed > 0 else 0)
            self._loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error("파이프라인 오류: %s (%s)", err, dbg)
            self._loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, d = msg.parse_warning()
            logger.warning("%s (%s)", w, d)
        return True

    def run(self):
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._bus_call)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("PLAYING 전환 실패")

        self._push_thread = threading.Thread(
            target=self._push_loop, daemon=True, name="jpeg-push"
        )
        self._push_thread.start()
        self._t_start = time.perf_counter()

        try:
            self._loop.run()
        except KeyboardInterrupt:
            logger.info("사용자 중단")
        finally:
            self._pipeline.set_state(Gst.State.NULL)
            if self._push_thread and self._push_thread.is_alive():
                self._push_thread.join(timeout=2.0)


# ══════════════════════════════════════════════════════════════════════════════
def _save_coco_result(path: str, coco_dets: list, img_files: list,
                      fname_to_imgid: dict, imgid_to_wh: dict,
                      categories: list, config: str) -> None:
    """탐지 결과를 instances_val2017.json 과 동일한 COCO 형식으로 저장.

    구조: info / images / annotations / categories
    annotation 1건 = detection 1건, bbox는 원본 이미지 좌표계 [x, y, w, h]
    """
    images = []
    for p in img_files:
        stem   = os.path.splitext(os.path.basename(p))[0]
        img_id = fname_to_imgid.get(stem)
        if img_id is None:
            continue
        w, h = imgid_to_wh.get(img_id, (MUX_W, MUX_H))
        images.append({
            "id":        img_id,
            "file_name": os.path.basename(p),
            "width":     w,
            "height":    h,
        })

    annotations = []
    for ann_id, det in enumerate(coco_dets, start=1):
        x, y, bw, bh = det["bbox"]
        annotations.append({
            "id":          ann_id,
            "image_id":    det["image_id"],
            "category_id": det["category_id"],
            "bbox":        det["bbox"],          # [x, y, w, h]  원본 좌표계
            "area":        round(bw * bh, 2),
            "score":       det["score"],
            "iscrowd":     0,
        })

    output = {
        "info": {
            "description":  "YOLOv8m GPU INT8 inference results",
            "config":       config,
            "version":      "1.0",
            "date_created": datetime.now().strftime("%Y/%m/%d"),
        },
        "images":      images,
        "annotations": annotations,
        "categories":  categories,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("COCO 형식 탐지 결과 저장: %s  (%d images, %d annotations)",
                path, len(images), len(annotations))


def run_coco_eval(ann_file: str, coco_dets: list, eval_img_ids: list | None = None) -> dict:
    """COCOeval 수행 후 주요 지표 dict 반환.

    eval_img_ids: None이면 전체 GT 기준, 지정 시 해당 이미지만 평가 (부분 실행 시 사용)
    """
    coco_gt = COCO(ann_file)

    if not coco_dets:
        logger.error("탐지 결과 없음 — 평가 불가")
        return {}

    coco_dt = coco_gt.loadRes(coco_dets)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if eval_img_ids is not None:
        ev.params.imgIds = eval_img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stats = ev.stats
    return {
        "AP@0.50:0.95": round(float(stats[0]), 4),
        "AP@0.50":      round(float(stats[1]), 4),
        "AP@0.75":      round(float(stats[2]), 4),
        "AP_small":     round(float(stats[3]), 4),
        "AP_medium":    round(float(stats[4]), 4),
        "AP_large":     round(float(stats[5]), 4),
        "AR@1":         round(float(stats[6]), 4),
        "AR@10":        round(float(stats[7]), 4),
        "AR@100":       round(float(stats[8]), 4),
        "AR_small":     round(float(stats[9]), 4),
        "AR_medium":    round(float(stats[10]), 4),
        "AR_large":     round(float(stats[11]), 4),
    }


def parse_args():
    p = argparse.ArgumentParser(description="GPU / DLA INT8 COCO mAP 측정")
    p.add_argument("--dla-core",   type=int, default=None,
                   help="가속기 선택: 0=DLA Core 0, 1=DLA Core 1, 미지정=GPU")
    p.add_argument("--config",     default=None,
                   help="nvinfer 설정 파일 직접 지정 (미지정 시 dla-core에 따라 자동 선택)")
    p.add_argument("--image-dir",  default=IMG_DIR)
    p.add_argument("--ann-file",   default=ANN_FILE)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--save-dets",  default=None, metavar="PATH",
                   help="탐지 결과를 instances_val2017.json 과 동일한 COCO 형식으로 저장")
    p.add_argument("--quiet",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quiet:
        _redirect_c_fds_to_devnull()

    Gst.init(None)

    # 가속기 선택
    if args.config:
        config     = args.config
        engine_key = "gpu" if args.dla_core not in (0, 1) else f"dla{args.dla_core}"
    elif args.dla_core == 0:
        config     = EVAL_CONFIGS["dla0"]
        engine_key = "dla0"
    elif args.dla_core == 1:
        config     = EVAL_CONFIGS["dla1"]
        engine_key = "dla1"
    else:
        config     = EVAL_CONFIGS["gpu"]
        engine_key = "gpu"

    engine_label = {"gpu": "GPU INT8", "dla0": "DLA Core 0 INT8", "dla1": "DLA Core 1 INT8"}[engine_key]
    result_path  = f"{BASE_DIR}/result_{engine_key}_map.json"

    logger.info("엔진: %s  config: %s", engine_label, config)

    logger.info("COCO 어노테이션 로드: %s", args.ann_file)
    yolo_to_coco_id, fname_to_imgid, imgid_to_wh, categories = _build_coco_maps(args.ann_file)
    logger.info("이미지 수: %d  카테고리 수: %d", len(fname_to_imgid), len(yolo_to_coco_id))

    img_files = _collect_images(args.image_dir, args.max_images)
    logger.info("추론 대상: %d장", len(img_files))

    pipe = EvalPipeline(config, img_files,
                        yolo_to_coco_id, fname_to_imgid, imgid_to_wh)
    pipe.run()

    logger.info("탐지 총 %d건 (%.1f건/장)", len(pipe.coco_dets),
                len(pipe.coco_dets) / max(pipe._proc_cnt, 1))

    if args.save_dets:
        _save_coco_result(args.save_dets, pipe.coco_dets, img_files,
                          fname_to_imgid, imgid_to_wh, categories, config)

    # --max-images 사용 시 평가 범위를 추론한 이미지로 제한
    eval_img_ids = None
    if args.max_images is not None:
        stems = [os.path.splitext(os.path.basename(p))[0] for p in img_files]
        eval_img_ids = [fname_to_imgid[s] for s in stems if s in fname_to_imgid]
        logger.info("평가 대상 image_id: %d개", len(eval_img_ids))

    logger.info("=" * 60)
    logger.info("COCO mAP 평가 시작")
    metrics = run_coco_eval(args.ann_file, pipe.coco_dets, eval_img_ids)

    if metrics:
        logger.info("=" * 60)
        logger.info("  %s  YOLOv8m  COCO val2017 mAP 결과", engine_label)
        logger.info("  AP@[0.50:0.95] = %.4f  (%.1f%%)", metrics["AP@0.50:0.95"], metrics["AP@0.50:0.95"] * 100)
        logger.info("  AP@0.50        = %.4f  (%.1f%%)", metrics["AP@0.50"],       metrics["AP@0.50"] * 100)
        logger.info("  AP@0.75        = %.4f  (%.1f%%)", metrics["AP@0.75"],       metrics["AP@0.75"] * 100)
        logger.info("  AP small       = %.4f", metrics["AP_small"])
        logger.info("  AP medium      = %.4f", metrics["AP_medium"])
        logger.info("  AP large       = %.4f", metrics["AP_large"])
        logger.info("=" * 60)

        with open(result_path, "w") as f:
            json.dump({"engine": engine_label, "images": pipe._proc_cnt,
                       "detections": len(pipe.coco_dets), **metrics}, f, indent=2)
        logger.info("결과 저장: %s", result_path)


if __name__ == "__main__":
    main()
