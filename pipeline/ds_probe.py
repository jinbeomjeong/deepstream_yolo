import time

import pyds
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from ds_tensor import get_tensor_from_user_meta, parse_layer, read_layer_arr, parse_yolo11
from ds_meta import add_obj_meta


def make_probe_callback(muxer_w, muxer_h, labels, conf_thresh, infer_w=640, infer_h=640):
    """
    pgie src pad 프로브 콜백을 생성해 반환한다.

    반환값: (probe_fn, state)
      - probe_fn : GStreamer pad probe 에 등록할 콜백
      - state    : {"frame_count", "t_start"} — 종료 후 FPS 계산에 사용
    """
    state = {
        "frame_count":  0,
        "t_start":      time.time(),
        "win_t":        time.time(),
        "win_frames":   0,
        "diag_printed": False,
    }

    def pgie_src_pad_probe(pad, info, u_data):
        """
        텐서 탐색 우선순위:
          1) batch_meta.batch_user_meta_list   ← primary GIE 배치 모드
          2) 각 frame_meta.frame_user_meta_list (Case A: 프레임별 독립 텐서)
          3) frame[0] 의 frame_user_meta_list   (Case B: 배치 전체가 frame 0 버퍼에)
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        # ── 1) 모든 frame_meta 수집 ─────────────────────────────────────────
        frames = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frames.append(pyds.NvDsFrameMeta.cast(l_frame.data))
            except StopIteration:
                break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        n_frames = len(frames)
        if n_frames == 0:
            return Gst.PadProbeReturn.OK

        tensors = [None] * n_frames
        diag_source = "none"

        # ── 2) batch_user_meta_list 우선 탐색 ───────────────────────────────
        layer = get_tensor_from_user_meta(batch_meta.batch_user_meta_list)
        if layer is not None:
            batch, shape = parse_layer(layer, n_frames)
            for i in range(n_frames):
                tensors[i] = batch[i]
            diag_source = f"batch_user_meta | inferDims={shape}"

        # ── 3) frame_user_meta_list 탐색 ────────────────────────────────────
        if tensors[0] is None:
            frame_layers = [get_tensor_from_user_meta(fm.frame_user_meta_list)
                            for fm in frames]
            n_found = sum(1 for x in frame_layers if x is not None)

            if n_found == n_frames:
                # 케이스 A: 모든 프레임에 독립 텐서
                for i, lyr in enumerate(frame_layers):
                    if lyr is None:
                        continue
                    dims  = lyr.inferDims
                    shape = [dims.d[j] for j in range(dims.numDims)]
                    n_elem = 1
                    for s in shape:
                        n_elem *= s
                    tensors[i] = read_layer_arr(lyr, n_elem).reshape(shape)
                diag_source = f"frame_user_meta Case-A | inferDims={shape}"

            elif n_found > 0:
                # 케이스 B: frame 0 버퍼에 배치 전체 저장
                first = next(i for i, x in enumerate(frame_layers) if x is not None)
                batch, shape = parse_layer(frame_layers[first], n_frames)
                for i in range(n_frames):
                    tensors[i] = batch[i]
                diag_source = (f"frame_user_meta Case-B "
                               f"({n_found}/{n_frames} frames have meta) | inferDims={shape}")

        # ── 4) 최초 1회 진단 출력 ───────────────────────────────────────────
        if not state["diag_printed"]:
            state["diag_printed"] = True
            print(f"[진단] n_frames={n_frames} | 텐서 출처={diag_source}")
            for i, fm in enumerate(frames):
                valid = tensors[i] is not None
                conf  = float(tensors[i][:, 4].max()) if valid else 0.0
                print(f"  [idx={i}] src={fm.source_id} batch_id={fm.batch_id} "
                      f"tensor={'유효' if valid else '없음'} max_conf={conf:.3f}")

        if all(t is None for t in tensors):
            return Gst.PadProbeReturn.OK

        # ── 5) 각 프레임에 detections 적용 ──────────────────────────────────
        for i, fm in enumerate(frames):
            if tensors[i] is None:
                continue

            dets = parse_yolo11(tensors[i], muxer_w, muxer_h, conf_thresh, infer_w, infer_h)

            state["frame_count"] += 1
            state["win_frames"]  += 1

            for det in dets:
                add_obj_meta(batch_meta, fm, det, labels)

            if state["frame_count"] % 100 == 0:
                now     = time.time()
                elapsed = now - state["win_t"]
                fps     = state["win_frames"] / elapsed if elapsed > 0 else 0
                print(f"[{state['frame_count']:6d}프레임] FPS={fps:.1f}")
                state["win_t"]      = now
                state["win_frames"] = 0

        return Gst.PadProbeReturn.OK

    return pgie_src_pad_probe, state
