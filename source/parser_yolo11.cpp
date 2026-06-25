#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include "nvdsinfer_custom_impl.h"

static bool getTensorLayout(
    NvDsInferDims const& dims,
    int channels,
    int& numAnchors,
    int& channelStride)
{
    for (int cDim = 0; cDim < dims.numDims; ++cDim) {
        if (dims.d[cDim] != channels) {
            continue;
        }

        int anchors = 1;
        for (int i = cDim + 1; i < dims.numDims; ++i) {
            anchors *= dims.d[i];
        }
        if (anchors <= 0) {
            continue;
        }

        int leading = 1;
        for (int i = 0; i < cDim; ++i) {
            leading *= dims.d[i];
        }
        if (leading != 1) {
            continue;
        }

        numAnchors = anchors;
        channelStride = anchors;
        return true;
    }

    return false;
}

static int findLayerByChannels(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    int channels)
{
    for (size_t i = 0; i < outputLayersInfo.size(); ++i) {
        NvDsInferDims const& dims = outputLayersInfo[i].inferDims;
        if (dims.numDims < 2) {
            continue;
        }

        for (int d = 0; d < dims.numDims - 1; ++d) {
            if (dims.d[d] == channels) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    int boxesLayerIndex = -1;
    int classesLayerIndex = -1;

    for (size_t i = 0; i < outputLayersInfo.size(); i++) {
        std::string layerName = outputLayersInfo[i].layerName
            ? outputLayersInfo[i].layerName
            : "";
        if (layerName == "output_boxes") {
            boxesLayerIndex = static_cast<int>(i);
        } else if (layerName == "output_classes") {
            classesLayerIndex = static_cast<int>(i);
        }
    }

    if (boxesLayerIndex == -1) {
        boxesLayerIndex = findLayerByChannels(outputLayersInfo, 4);
    }
    if (classesLayerIndex == -1) {
        classesLayerIndex = findLayerByChannels(outputLayersInfo, 80);
    }

    if (boxesLayerIndex == -1 || classesLayerIndex == -1) {
        std::cerr << "Error: output_boxes/output_classes 레이어를 찾을 수 없습니다."
                  << std::endl;
        return false;
    }

    NvDsInferLayerInfo const& boxesLayer = outputLayersInfo[boxesLayerIndex];
    NvDsInferLayerInfo const& classesLayer = outputLayersInfo[classesLayerIndex];
    NvDsInferDims const& boxesDims = boxesLayer.inferDims;
    NvDsInferDims const& classesDims = classesLayer.inferDims;

    if (boxesDims.numDims < 2 || classesDims.numDims < 2) {
        std::cerr << "Error: 출력 텐서 차원이 올바르지 않습니다." << std::endl;
        return false;
    }

    int numAnchors = 0;
    int boxesStride = 0;
    int classesAnchors = 0;
    int classesStride = 0;
    if (!getTensorLayout(boxesDims, 4, numAnchors, boxesStride) ||
        !getTensorLayout(classesDims, 80, classesAnchors, classesStride)) {
        std::cerr << "Error: 출력 텐서 layout이 예상과 다릅니다." << std::endl;
        return false;
    }
    if (numAnchors != classesAnchors) {
        std::cerr << "Error: boxes/classes anchor 수가 다릅니다: "
                  << numAnchors << " vs " << classesAnchors << std::endl;
        return false;
    }

    const float* boxes = static_cast<const float*>(boxesLayer.buffer);
    const float* classes = static_cast<const float*>(classesLayer.buffer);

    for (int i = 0; i < numAnchors; i++) {
        float max_prob = 0.0f;
        int max_class_id = -1;

        for (int c = 0; c < 80; c++) {
            float prob = classes[c * classesStride + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_class_id = c;
            }
        }

        if (max_class_id < 0) {
            continue;
        }

        if (max_prob >= detectionParams.perClassPreclusterThreshold[max_class_id]) {
            float cx = boxes[0 * boxesStride + i];
            float cy = boxes[1 * boxesStride + i];
            float w  = boxes[2 * boxesStride + i];
            float h  = boxes[3 * boxesStride + i];

            float left = std::max(0.0f, cx - (w / 2.0f));
            float top = std::max(0.0f, cy - (h / 2.0f));
            float right = std::min(static_cast<float>(networkInfo.width), cx + (w / 2.0f));
            float bottom = std::min(static_cast<float>(networkInfo.height), cy + (h / 2.0f));

            NvDsInferParseObjectInfo obj;
            obj.classId = max_class_id;
            obj.detectionConfidence = max_prob;
            obj.left = left;
            obj.top = top;
            obj.width = std::max(0.0f, right - left);
            obj.height = std::max(0.0f, bottom - top);

            objectList.push_back(obj);
        }
    }
    return true;
}
