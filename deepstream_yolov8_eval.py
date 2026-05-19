#!/usr/bin/env python3
"""
deepstream_yolov8_eval.py
=========================
val2017 이미지를 4채널(동일 이미지)로 추론하고
채널 0의 탐지 결과(클래스 번호, 컨피던스, 바운딩 박스)를 JSON으로 저장.

저장된 결과는 COCO 레이블과 비교하여 클래스 정확도 및 바운딩 박스 정확도 분석에 사용.
바운딩 박스는 nvstreammux 해상도(MUXER_W × MUXER_H) 기준 절대 픽셀 [left, top, width, height].

[파이프라인]
  multifilesrc(ch0) ─┐
  multifilesrc(ch1) ─┴→ nvstreammux_dla0(batch=2) → nvinfer_dla0 ─┐
                                                                    funnel → fakesink
  multifilesrc(ch2) ─┐
  multifilesrc(ch3) ─┴→ nvstreammux_dla1(batch=2) → nvinfer_dla1 ─┘

채널 0 (DLA Core 0, source_id=0) 의 탐지 결과만 저장.

[실행]
  python3 deepstream_yolov8_eval.py
  python3 deepstream_yolov8_eval.py --max-images 500
  python3 deepstream_yolov8_eval.py --image-dir /data/val2017 --output det.json
"""

import sys
import os
import glob
import json
import time
import logging
import argparse
import tempfile
import shutil
import threading

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst, GLib
import pyds


# ══════════════════════════════════════════════════════════════════════════════
# 로거
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ds_eval")


# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════
PGIE_CONFIG_DLA0 = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolov8_dla0_int8.txt"
PGIE_CONFIG_DLA1 = "/home/nvidia/workspace/deepstream_yolo/config_infer_yolov8_dla1_int8.txt"

MUXER_W   = 1920
MUXER_H   = 1080
DLA_BATCH = 2

DLA_GROUPS = {
    0: [0, 1],
    1: [2, 3],
}


# ══════════════════════════════════════════════════════════════════════════════
# 탐지 결과 저장소 (probe 콜백 → 메인 스레드 간 공유)
# ══════════════════════════════════════════════════════════════════════════════
_detections: dict[int, list] = {}   # frame_idx → [{class_id, confidence, bbox_ltwh}]
_lock = threading.Lock()
_image_list: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# 인자 파싱
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepStream YOLOv8m 4채널 DLA 추론 결과 저장 (val2017 평가용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "실행 예시:\n"
            "  python3 deepstream_yolov8_eval.py\n"
            "  python3 deepstream_yolov8_eval.py --max-images 500\n"
            "  python3 deepstream_yolov8_eval.py --image-dir /data/val2017 --output det.json\n"
        ),
    )
    parser.add_argument("--image-dir", default="/home/nvidia/workspace/val2017",
                        help="이미지 디렉터리 (기본값: /home/nvidia/workspace/val2017)")
    parser.add_argument("--output", default="detections.json",
                        help="저장할 JSON 파일 경로 (기본값: detections.json)")
    parser.add_argument("--max-images", type=int, default=None, metavar="N",
                        help="처리할 최대 이미지 수 (기본값: 전체)")
    return parser.parse_args()


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
        logger.info("EOS — 모든 이미지 처리 완료")
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
# 이미지 소스 → nvstreammux
# ══════════════════════════════════════════════════════════════════════════════
def add_image_source_to_mux(pipeline: Gst.Pipeline,
                             ch_idx: int,
                             location_pattern: str,
                             n_images: int,
                             mux: Gst.Element,
                             mux_sink_idx: int) -> None:
    """
    multifilesrc → jpegdec → nvvideoconvert → NVMM → nvstreammux

    location_pattern: multifilesrc 용 printf 패턴 (예: /tmp/eval/000000_%06d.jpg)
    n_images        : 처리할 이미지 수 (stop-index = n_images - 1)
    """
    src = make_element("multifilesrc", f"src-ch{ch_idx}")
    src.set_property("location",   location_pattern)
    src.set_property("index",      0)
    src.set_property("stop-index", n_images - 1)
    src.set_property("loop",       False)

    src_caps = make_element("capsfilter", f"caps-src-ch{ch_idx}")
    src_caps.set_property("caps", Gst.Caps.from_string("image/jpeg,framerate=1/1"))

    decoder = make_element("jpegdec", f"jpegdec-ch{ch_idx}")

    conv = make_element("nvvideoconvert", f"conv-ch{ch_idx}")

    nvmm_caps = make_element("capsfilter", f"caps-nvmm-ch{ch_idx}")
    nvmm_caps.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=NV12"
    ))

    for el in (src, src_caps, decoder, conv, nvmm_caps):
        pipeline.add(el)

    for a, b in [(src, src_caps), (src_caps, decoder),
                 (decoder, conv), (conv, nvmm_caps)]:
        if not a.link(b):
            raise RuntimeError(f"{a.get_name()} → {b.get_name()} 링크 실패")

    sink_pad = mux.request_pad_simple(f"sink_{mux_sink_idx}")
    if not sink_pad:
        raise RuntimeError(f"mux sink_{mux_sink_idx} 패드 요청 실패 (채널 {ch_idx})")

    ret = nvmm_caps.get_static_pad("src").link(sink_pad)
    if ret != Gst.PadLinkReturn.OK:
        raise RuntimeError(
            f"caps-nvmm-ch{ch_idx} → {mux.get_name()} sink_{mux_sink_idx} 링크 실패: {ret}"
        )

    logger.info("채널 %d → %s sink_%d 연결 완료", ch_idx, mux.get_name(), mux_sink_idx)


# ══════════════════════════════════════════════════════════════════════════════
# 파이프라인 구성
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(location_pattern: str, n_images: int) -> tuple:
    Gst.init(None)
    pipeline = Gst.Pipeline()
    loop = GLib.MainLoop()

    pgie_configs = {0: PGIE_CONFIG_DLA0, 1: PGIE_CONFIG_DLA1}
    nvinfer_outputs = []

    for dla_core, channels in DLA_GROUPS.items():
        mux = make_element("nvstreammux", f"mux-dla{dla_core}")
        mux.set_property("width",                MUXER_W)
        mux.set_property("height",               MUXER_H)
        mux.set_property("batch-size",           DLA_BATCH)
        mux.set_property("batched-push-timeout", 4_000_000)   # 4초 — 이미지 소스 비실시간
        mux.set_property("nvbuf-memory-type",    4)
        mux.set_property("live-source",          0)            # 비실시간 모드
        pipeline.add(mux)

        for ch_idx in channels:
            add_image_source_to_mux(
                pipeline, ch_idx, location_pattern, n_images, mux, ch_idx
            )

        pgie = make_element("nvinfer", f"pgie-dla{dla_core}")
        pgie.set_property("config-file-path", pgie_configs[dla_core])
        pipeline.add(pgie)

        if not mux.link(pgie):
            raise RuntimeError(f"mux-dla{dla_core} → pgie-dla{dla_core} 링크 실패")

        nvinfer_outputs.append(pgie)
        logger.info("mux-dla%d → pgie-dla%d 연결 완료 (설정: %s)",
                    dla_core, dla_core, pgie_configs[dla_core])

    funnel = make_element("funnel", "funnel")
    pipeline.add(funnel)
    for pgie in nvinfer_outputs:
        if not pgie.link(funnel):
            raise RuntimeError(f"{pgie.get_name()} → funnel 링크 실패")

    sink = make_element("fakesink", "sink")
    sink.set_property("sync", False)
    pipeline.add(sink)
    if not funnel.link(sink):
        raise RuntimeError("funnel → fakesink 링크 실패")

    return pipeline, loop


# ══════════════════════════════════════════════════════════════════════════════
# 탐지 결과 probe  (pgie-dla0, source_id=0 만 수집)
# ══════════════════════════════════════════════════════════════════════════════
def attach_detection_probe(pipeline: Gst.Pipeline) -> None:
    def probe_cb(pad, info):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        frame_list = batch_meta.frame_meta_list
        while frame_list:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            if frame_meta.source_id == 0:
                frame_idx = frame_meta.frame_num
                dets = []

                obj_list = frame_meta.obj_meta_list
                while obj_list:
                    try:
                        obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                        r = obj.rect_params
                        dets.append({
                            "class_id":   int(obj.class_id),
                            "confidence": round(float(obj.confidence), 4),
                            "bbox_ltwh":  [
                                round(float(r.left),   2),
                                round(float(r.top),    2),
                                round(float(r.width),  2),
                                round(float(r.height), 2),
                            ],
                        })
                        obj_list = obj_list.next
                    except StopIteration:
                        break

                with _lock:
                    _detections[frame_idx] = dets

                if frame_idx % 100 == 0:
                    logger.info("진행 중: 프레임 %d / %d  탐지 %d개",
                                frame_idx, len(_image_list), len(dets))

            try:
                frame_list = frame_list.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    pgie = pipeline.get_by_name("pgie-dla0")
    if pgie:
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_cb)
        logger.info("탐지 probe 부착: pgie-dla0 (source_id=0 만 저장)")


# ══════════════════════════════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════════════════════════════
def save_results(output_path: str) -> None:
    results = []
    for frame_idx in sorted(_detections):
        if frame_idx >= len(_image_list):
            continue
        img_file = os.path.basename(_image_list[frame_idx])
        try:
            image_id = int(os.path.splitext(img_file)[0])
        except ValueError:
            image_id = frame_idx

        results.append({
            "image_file": img_file,
            "image_id":   image_id,
            "detections": _detections[frame_idx],
        })

    output = {
        "meta": {
            "muxer_width":    MUXER_W,
            "muxer_height":   MUXER_H,
            "source_channel": 0,
            "bbox_format":    "left, top, width, height  (muxer 해상도 기준 절대 픽셀)",
            "total_images":   len(results),
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_dets = sum(len(r["detections"]) for r in results)
    logger.info("결과 저장: %s", output_path)
    logger.info("  이미지 %d개  탐지 %d개  (평균 %.1f개/이미지)",
                len(results), total_dets, total_dets / max(len(results), 1))


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _image_list
    args = parse_args()

    # 이미지 목록 구성 (정렬 → frame_num 과 1:1 대응)
    images = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not images:
        logger.error("이미지 없음: %s", args.image_dir)
        sys.exit(1)
    if args.max_images:
        images = images[:args.max_images]
    _image_list = images

    logger.info("=" * 60)
    logger.info("DeepStream YOLOv8m 4채널 DLA 평가 파이프라인")
    logger.info("=" * 60)
    logger.info("이미지 디렉터리: %s", args.image_dir)
    logger.info("처리 이미지 수 : %d", len(images))
    logger.info("출력 파일      : %s", args.output)
    logger.info("muxer 해상도   : %d × %d", MUXER_W, MUXER_H)

    # multifilesrc 는 printf 패턴(%06d) 을 요구하므로
    # 임시 디렉터리에 순번 심볼릭 링크 생성
    tmp_dir = tempfile.mkdtemp(prefix="ds_eval_")
    logger.info("임시 디렉터리  : %s  (종료 시 자동 삭제)", tmp_dir)

    try:
        for i, img_path in enumerate(images):
            os.symlink(img_path, os.path.join(tmp_dir, f"{i:06d}.jpg"))
        location_pattern = os.path.join(tmp_dir, "%06d.jpg")

        pipeline, loop = build_pipeline(location_pattern, len(images))
        attach_detection_probe(pipeline)

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        logger.info("파이프라인 시작...")
        t_start = time.perf_counter()
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("파이프라인 PLAYING 전환 실패")
            pipeline.set_state(Gst.State.NULL)
            sys.exit(1)

        try:
            loop.run()
        except KeyboardInterrupt:
            logger.info("사용자 중단 (Ctrl+C)")
        finally:
            elapsed = time.perf_counter() - t_start
            logger.info("소요 시간: %.1f초  (%.1f 이미지/초)",
                        elapsed, len(images) / max(elapsed, 1e-6))
            pipeline.set_state(Gst.State.NULL)

        save_results(args.output)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("임시 디렉터리 삭제 완료: %s", tmp_dir)


if __name__ == "__main__":
    main()
