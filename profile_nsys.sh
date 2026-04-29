#!/usr/bin/env bash
# profile_nsys.sh — Nsight Systems profiling for DeepStream 4ch DLA pipeline
#
# Usage:
#   ./profile_nsys.sh [video_path] [--display]
#
# Output:
#   profile_dla_<timestamp>.nsys-rep  (open with Nsight Systems GUI)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${SCRIPT_DIR}/profile_dla_${TIMESTAMP}"

VIDEO="${1:-/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4}"
EXTRA_ARGS="${2:-}"

# Profiling duration (seconds) — enough to reach steady-state after engine load
DURATION=30

echo "================================================================"
echo " Nsight Systems — DeepStream YOLOv8m 4ch DLA Profiling"
echo "================================================================"
echo " Video      : ${VIDEO}"
echo " Duration   : ${DURATION}s (after pipeline steady-state)"
echo " Output     : ${OUTPUT}.nsys-rep"
echo "================================================================"

VENV_DIR="/home/nvidia/workspace/arround_view/venv"
VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_SITE="${VENV_DIR}/lib/python3.10/site-packages"

# venv 활성화
source "${VENV_DIR}/bin/activate"

# nsys가 fork한 프로세스에도 site-packages가 보이도록 PYTHONPATH 명시
export PYTHONPATH="${VENV_SITE}${PYTHONPATH:+:${PYTHONPATH}}"

echo " Python     : ${VENV_PYTHON}"
echo " PYTHONPATH : ${PYTHONPATH}"
echo "================================================================"

nsys profile \
    --trace=cuda,nvtx,cudla,osrt,tegra-accelerators \
    --accelerator-trace=tegra-accelerators \
    --gpu-metrics-device=all \
    --gpu-metrics-frequency=10000 \
    --cuda-memory-usage=true \
    --backtrace=fp \
    --duration=${DURATION} \
    --output="${OUTPUT}" \
    --force-overwrite=true \
    --export=sqlite \
    "${VENV_PYTHON}" "${SCRIPT_DIR}/deepstream_yolov8_4ch_dla.py" "${VIDEO}" ${EXTRA_ARGS}

echo ""
echo "================================================================"
echo " 프로파일링 완료"
echo " 리포트 파일: ${OUTPUT}.nsys-rep"
echo ""
echo " 분석 방법:"
echo "  1. [GUI]  Nsight Systems 앱에서 .nsys-rep 파일 열기"
echo "  2. [CLI]  nsys analyze ${OUTPUT}.nsys-rep"
echo "  3. [SQL]  sqlite3 ${OUTPUT}.sqlite 로 직접 쿼리"
echo "================================================================"
