// parser_yolo11.cpp
// YOLO 탐지 결과 파서 — split 출력 (output_boxes / output_classes) 지원
//
// 입력 텐서:
//   output_boxes   [4,  N] : cx, cy, w, h  (픽셀 좌표)
//   output_classes [80, N] : 클래스 확률  (sigmoid 적용)
//
// split 출력을 사용하면 INT8 스케일이 각 텐서에 독립 적용되어
// 좌표(~512)와 확률(0~1)의 스케일 충돌이 제거됨.

#include <iostream>
#include <vector>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolov8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo  const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const float* boxes_buf   = nullptr;  // [4,  N]
    const float* classes_buf = nullptr;  // [80, N]
    int num_anchors = 0;
    int num_classes = 0;

    for (const auto& layer : outputLayersInfo) {
        auto& d = layer.inferDims;
        // 배치 차원 포함(numDims==3) / 미포함(numDims==2) 양쪽 처리
        int ch = (d.numDims == 3) ? d.d[1] : d.d[0];
        int an = (d.numDims == 3) ? d.d[2] : d.d[1];
        if (ch == 4) {
            boxes_buf   = (const float*)layer.buffer;
            num_anchors = an;
        } else if (ch > 4) {
            classes_buf = (const float*)layer.buffer;
            num_classes = ch;
        }
    }

    if (!boxes_buf || !classes_buf) {
        std::cerr << "[Parser] output_boxes / output_classes 를 찾을 수 없습니다." << std::endl;
        return false;
    }

    for (int i = 0; i < num_anchors; i++) {
        float max_prob = 0.0f;
        int   max_cls  = -1;
        for (int c = 0; c < num_classes; c++) {
            float p = classes_buf[c * num_anchors + i];
            if (p > max_prob) { max_prob = p; max_cls = c; }
        }
        if (max_cls < 0) continue;
        if (max_prob < detectionParams.perClassPreclusterThreshold[max_cls]) continue;

        float cx = boxes_buf[0 * num_anchors + i];
        float cy = boxes_buf[1 * num_anchors + i];
        float w  = boxes_buf[2 * num_anchors + i];
        float h  = boxes_buf[3 * num_anchors + i];

        NvDsInferParseObjectInfo obj;
        obj.classId            = max_cls;
        obj.detectionConfidence = max_prob;
        obj.left   = cx - w * 0.5f;
        obj.top    = cy - h * 0.5f;
        obj.width  = w;
        obj.height = h;
        objectList.push_back(obj);
    }
    // 처음 3회만 탐지 수 출력
    static int _cnt = 0;
    if (++_cnt <= 3)
        fprintf(stderr, "[Parser] call#%d  detections=%zu\n", _cnt, objectList.size());

    return true;
}
