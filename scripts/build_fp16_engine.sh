#!/usr/bin/env bash
# DeepStream YOLO11m GPU FP16 파이프라인 — ONNX 내보내기 + TensorRT 엔진 빌드
# 대상: deepstream_yolo11_4ch_gpu_fp16.py  (프로젝트 루트)

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPTS_DIR")"
VENV="/home/nvidia/workspace/arround_view/venv/bin/activate"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

MODELS_DIR="$PROJECT_ROOT/models"
ONNX_FILE="$MODELS_DIR/yolo11m.onnx"
ENGINE_FILE="$MODELS_DIR/yolo11m.onnx_b4_gpu_fp16.engine"
TIMING_CACHE="$MODELS_DIR/yolo11_timing.cache"

mkdir -p "$MODELS_DIR"

echo "======================================================"
echo " DeepStream YOLO11m GPU FP16 엔진 빌드"
echo " 프로젝트 루트: $PROJECT_ROOT"
echo "======================================================"

# ── Step 1: venv 확인 및 활성화 ───────────────────────────────────────────────
if [ ! -f "$VENV" ]; then
    echo "[오류] venv를 찾을 수 없습니다: $VENV"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV"
echo "[1/2] venv 활성화 완료"

# ── Step 2: ONNX 내보내기 (batch=4, nms=True, opset=17) ───────────────────────
if [ -f "$ONNX_FILE" ]; then
    echo "[1/2] ONNX 건너뜀 — 이미 존재: models/$(basename "$ONNX_FILE")"
else
    echo "[1/2] YOLO11m → ONNX 내보내기 시작..."
    echo "      (yolo11m.pt 없으면 자동 다운로드)"
    cd "$SCRIPTS_DIR"
    python3 export_yolo11.py
    if [ ! -f "$ONNX_FILE" ]; then
        echo "[오류] ONNX 파일이 생성되지 않았습니다: $ONNX_FILE"
        exit 1
    fi
    echo "[1/2] ONNX 저장 완료: models/$(basename "$ONNX_FILE")"
fi

# ── Step 3: TensorRT FP16 엔진 빌드 ──────────────────────────────────────────
if [ -f "$ENGINE_FILE" ]; then
    echo "[2/2] 엔진 건너뜀 — 이미 존재: models/$(basename "$ENGINE_FILE")"
else
    if [ ! -f "$TRTEXEC" ]; then
        echo "[오류] trtexec를 찾을 수 없습니다: $TRTEXEC"
        exit 1
    fi

    echo "[2/2] TensorRT FP16 엔진 빌드 시작 (수 분 소요)..."
    "$TRTEXEC" \
        --onnx="$ONNX_FILE"               \
        --saveEngine="$ENGINE_FILE"       \
        --fp16                            \
        --useCudaGraph                    \
        --timingCacheFile="$TIMING_CACHE" \
        --avgTiming=100                   \
        --sparsity=enable                 \
        --memPoolSize=workspace:8192MiB

    if [ ! -f "$ENGINE_FILE" ]; then
        echo "[오류] 엔진 파일이 생성되지 않았습니다: $ENGINE_FILE"
        exit 1
    fi
    echo "[2/2] 엔진 저장 완료: models/$(basename "$ENGINE_FILE")"
fi

# ── 완료 메시지 ───────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " 빌드 완료"
echo "   ONNX   : $ONNX_FILE"
echo "   Engine : $ENGINE_FILE"
echo "======================================================"
echo ""
echo " 실행 방법:"
echo "   source $VENV"
echo "   cd $PROJECT_ROOT"
echo ""
echo "   # 헤드리스 (추론만)"
echo "   python3 deepstream_yolo11_4ch_gpu_fp16.py"
echo ""
echo "   # 화면 출력"
echo "   DISPLAY=:0 python3 deepstream_yolo11_4ch_gpu_fp16.py"
echo ""
echo "   # 영상 저장"
echo "   python3 deepstream_yolo11_4ch_gpu_fp16.py -o out.mp4"
echo ""
echo "   # 화면 출력 + 저장"
echo "   DISPLAY=:0 python3 deepstream_yolo11_4ch_gpu_fp16.py -o out.mp4"
