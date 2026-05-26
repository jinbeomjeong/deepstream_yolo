// parser_yolo11.cpp
// YOLO11m 커스텀 바운딩박스 파서 — NMS 내장 엔진 출력 [300, 6]
//
// 입력 텐서 (레이어 1개):
//   shape   : [300, 6]  (numDims=2) 또는 [1, 300, 6]  (numDims=3)
//   col 0-3 : x1, y1, x2, y2  (640×640 letterbox 기준 픽셀 좌표)
//   col 4   : confidence  (NMS 후 최종 확률, 0~1)
//   col 5   : class_id    (float → int 변환)
//
// DeepStream nvinfer 가 NvDsInferParseObjectInfo 좌표를
// 실제 프레임 해상도로 자동 역변환 (maintain-aspect-ratio=1 적용).
//
// config 설정:
//   custom-lib-path      = .../pipeline/lib_parser_yolo11.so
//   parse-bbox-func-name = NvDsInferParseYolo11
//   network-type         = 0   (Detector — parse-bbox-func-name 호출 조건)
//   cluster-mode         = 4   (NMS 중복 방지 — 엔진에서 이미 처리)
//   num-detected-classes = 80
//   [class-attrs-all]
//   pre-cluster-threshold = 0.30

#include <cstdio>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"

static const int FIELDS = 6;  // [x1, y1, x2, y2, conf, class_id]

extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo>  const& outputLayersInfo,
    NvDsInferNetworkInfo             const& networkInfo,
    NvDsInferParseDetectionParams    const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    const float* data    = nullptr;
    int          num_det = 0;

    // 출력 레이어에서 [300, 6] 형태 탐색 (최초 1회 전체 레이어 형태 진단 출력)
    static bool s_layer_printed = false;
    for (const auto& layer : outputLayersInfo) {
        const auto& d = layer.inferDims;
        if (!s_layer_printed) {
            fprintf(stderr, "[NvDsInferParseYolo11] layer='%s' numDims=%d",
                    layer.layerName, d.numDims);
            for (int k = 0; k < d.numDims; k++)
                fprintf(stderr, " d[%d]=%d", k, d.d[k]);
            fprintf(stderr, "\n");
        }
        if (d.numDims == 2 && d.d[1] == FIELDS) {
            // [300, 6]
            data    = static_cast<const float*>(layer.buffer);
            num_det = d.d[0];
            break;
        } else if (d.numDims == 3 && d.d[2] == FIELDS) {
            // [1, 300, 6]
            data    = static_cast<const float*>(layer.buffer);
            num_det = d.d[1];
            break;
        }
    }
    s_layer_printed = true;

    if (!data) {
        std::cerr << "[NvDsInferParseYolo11] 출력 레이어를 찾을 수 없습니다 "
                     "(expected last-dim=6, got shapes above)." << std::endl;
        return false;
    }

    const int   num_classes   = static_cast<int>(detectionParams.numClassesConfigured);
    const int   thresh_size   = static_cast<int>(detectionParams.perClassPreclusterThreshold.size());
    const float default_thresh = 0.30f;   // [class-attrs-all] 미설정 시 폴백

    for (int i = 0; i < num_det; i++) {
        const float* row  = data + i * FIELDS;
        const float  conf = row[4];
        const int    cls  = static_cast<int>(row[5]);

        if (cls < 0 || cls >= num_classes) continue;

        // DS7.0 cluster-mode=4 환경에서 벡터 크기가 0일 수 있으므로 bounds 확인
        const float thresh = (cls < thresh_size)
                             ? detectionParams.perClassPreclusterThreshold[cls]
                             : default_thresh;
        if (conf < thresh) continue;

        const float x1 = row[0];
        const float y1 = row[1];
        const float x2 = row[2];
        const float y2 = row[3];

        // 유효하지 않은 박스 제거
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsInferObjectDetectionInfo obj{};
        obj.classId             = static_cast<unsigned int>(cls);
        obj.detectionConfidence = conf;
        obj.left                = x1;
        obj.top                 = y1;
        obj.width               = x2 - x1;
        obj.height              = y2 - y1;
        objectList.push_back(obj);
    }

    // 처음 3회만 탐지 수 출력
    static int s_call = 0;
    if (++s_call <= 3)
        fprintf(stderr, "[NvDsInferParseYolo11] call#%d  detections=%zu\n",
                s_call, objectList.size());

    return true;
}
