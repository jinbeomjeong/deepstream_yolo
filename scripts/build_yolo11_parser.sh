#!/usr/bin/env bash
# pipeline/parser_yolo11.cpp → pipeline/lib_parser_yolo11.so 빌드

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"

SRC="$PROJECT_ROOT/pipeline/parser_yolo11.cpp"
OUT="$PROJECT_ROOT/pipeline/lib_parser_yolo11.so"
DS_INCLUDES="/opt/nvidia/deepstream/deepstream/sources/includes"
CUDA_INCLUDES="/usr/local/cuda/include"

echo "======================================================"
echo " YOLO11m 커스텀 파서 빌드"
echo " 입력: $SRC"
echo " 출력: $OUT"
echo "======================================================"

if [ ! -f "$DS_INCLUDES/nvdsinfer_custom_impl.h" ]; then
    echo "[오류] DeepStream 헤더를 찾을 수 없습니다: $DS_INCLUDES"
    exit 1
fi

g++ -o "$OUT" "$SRC" \
    -Wall -std=c++14 -shared -fPIC -O2 \
    -I "$DS_INCLUDES" \
    -I "$CUDA_INCLUDES"

echo "[완료] $OUT"
