#!/bin/bash
# GPU 사용률 비교 측정: tegrastats 로깅 + DeepStream 추론 동시 실행
# 사용법: bash measure_gpu_util.sh [gpu|dla0|dla1]

ENGINE=${1:-gpu}
LOG_FILE="tegrastats_${ENGINE}.log"

case $ENGINE in
  gpu)  DLA_ARG="--dla-core 2" ;;
  dla0) DLA_ARG="--dla-core 0" ;;
  dla1) DLA_ARG="--dla-core 1" ;;
  *) echo "사용법: $0 [gpu|dla0|dla1]"; exit 1 ;;
esac

source /home/nvidia/workspace/arround_view/venv/bin/activate

echo "[$ENGINE] tegrastats 로깅 시작 → $LOG_FILE"
tegrastats --interval 500 > "$LOG_FILE" 2>/dev/null &
TSTAT_PID=$!
sleep 2   # 베이스라인 2초 수집

echo "[$ENGINE] 추론 시작..."
python3 deepstream_yolov8_image.py $DLA_ARG --quiet 2>&1 | grep -E "EOS|FPS"

sleep 1
kill $TSTAT_PID 2>/dev/null
echo "[$ENGINE] 완료 → $LOG_FILE ($(wc -l < $LOG_FILE)줄)"
