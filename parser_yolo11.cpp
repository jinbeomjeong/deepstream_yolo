// 파일명: custom_parser_yolo11.cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "nvdsinfer_custom_impl.h" // DeepStream 기본 헤더

extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // 1. 모델의 출력 레이어 찾기
    int layerIndex = -1;
    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        // 차원이 2차원 또는 3차원인 레이어를 출력 레이어로 간주
        if (outputLayersInfo[i].inferDims.numDims == 2 || outputLayersInfo[i].inferDims.numDims == 3) {
            layerIndex = i;
            break;
        }
    }
    
    if (layerIndex == -1) {
        std::cerr << "Error: 출력 레이어를 찾을 수 없습니다." << std::endl;
        return false;
    }

    // 2. 텐서 데이터 가져오기
    const float* output = (const float*)outputLayersInfo[layerIndex].buffer;
    auto dims = outputLayersInfo[layerIndex].inferDims;

    // 3. 차원 분석 (보통 [1, 84, 8400] 또는 [84, 8400])
    int dim_idx_classes = (dims.numDims == 3) ? 1 : 0;
    int dim_idx_anchors = (dims.numDims == 3) ? 2 : 1;
    
    int num_classes_plus_4 = dims.d[dim_idx_classes]; // 84 (4개의 좌표 + 80개의 클래스)
    int num_anchors = dims.d[dim_idx_anchors];       // 8400 (예측된 박스 개수)
    int num_classes = num_classes_plus_4 - 4;        // 80

    // 4. 8400개의 예측 결과를 하나씩 순회하며 분석
    for (int i = 0; i < num_anchors; i++) {
        float max_prob = 0.0f;
        int max_class_id = -1;

        // 가장 확률이 높은 클래스 찾기
        for (int c = 0; c < num_classes; c++) {
            // Tensor 메모리는 1차원 배열로 펼쳐져 있음 (Row-major)
            float prob = output[(4 + c) * num_anchors + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_class_id = c;
            }
        }

        // 5. 확률이 설정한 임계값(Threshold) 이상인 경우만 박스 추출
        if (max_prob >= detectionParams.perClassPreclusterThreshold[max_class_id]) {
            float cx = output[0 * num_anchors + i]; // 중심 X
            float cy = output[1 * num_anchors + i]; // 중심 Y
            float w  = output[2 * num_anchors + i]; // 너비
            float h  = output[3 * num_anchors + i]; // 높이

            NvDsInferParseObjectInfo obj;
            obj.classId = max_class_id;
            obj.detectionConfidence = max_prob;

            // DeepStream은 좌상단(Left, Top) 좌표와 너비, 높이를 요구함
            obj.left = cx - (w / 2.0f);
            obj.top = cy - (h / 2.0f);
            obj.width = w;
            obj.height = h;

            objectList.push_back(obj);
        }
    }
    return true;
}