import pyds


def add_obj_meta(batch_meta, frame_meta, det, labels):
    """detection 결과 하나를 NvDsObjectMeta 로 변환해 frame_meta 에 추가한다."""
    obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj.class_id   = det["class_id"]
    obj.confidence = det["conf"]
    obj.object_id  = 0xFFFFFFFFFFFFFFFF   # UNTRACKED

    c = obj.detector_bbox_info.org_bbox_coords
    c.left, c.top = det["x1"], det["y1"]
    c.width  = det["x2"] - det["x1"]
    c.height = det["y2"] - det["y1"]

    r = obj.rect_params
    r.left, r.top   = det["x1"], det["y1"]
    r.width, r.height = c.width, c.height
    r.border_width = 2
    r.border_color.set(0.0, 1.0, 0.0, 1.0)

    label = labels[det["class_id"]] if det["class_id"] < len(labels) else str(det["class_id"])
    t = obj.text_params
    t.display_text = f"{label} {det['conf']:.2f}"
    t.x_offset, t.y_offset = int(det["x1"]), max(0, int(det["y1"]) - 20)
    t.font_params.font_name = "Serif"
    t.font_params.font_size = 12
    t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    t.set_bg_clr = 1
    t.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)
