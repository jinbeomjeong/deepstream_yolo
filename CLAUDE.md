# CLAUDE.md — DeepStream YOLO11m 프로젝트 컨텍스트

## 사용자 프로필

Jetson 기반 엣지 AI 개발자. DeepStream + TensorRT + YOLO 파이프라인을 직접 구축·최적화.  
Python(GStreamer/pyds), C++(nvinfer 커스텀 파서), ONNX/TensorRT 변환 모두 익숙함.  
불필요한 코드(통계 로그, probe 오버헤드)를 제거해 성능을 높이는 것을 선호함.

## 프로젝트 구조 (FP16 GPU 파이프라인)

```
/home/nvidia/workspace/deepstream_yolo/
│
├── deepstream_yolo11_4ch_gpu_fp16.py     # 메인 파이프라인 (DISPLAY 환경변수로 화면 제어)
├── deepstream_yolo11_power_log.py        # 전력 측정 + CSV (--display 플래그)
├── deepstream_yolo11_frame_latency.py    # PGIE 배치 처리 시간 + CSV (--display 플래그)
│
├── pipeline/
│   ├── parser_yolo11.cpp                 # C++ 커스텀 파서 ([300,6] NMS 출력)
│   ├── lib_parser_yolo11.so              # 빌드 결과물
│   ├── ds_pipeline.py                    # GStreamer 헬퍼 (probe_video_size, bus_call,
│   │                                     #   make_src_and_connect, link_save_branch)
│   └── ds_labels.py / ds_meta.py / ds_probe.py / ds_tensor.py  # 구 Python 파서 (미사용)
│
├── config/
│   ├── config_infer_yolo11_gpu_fp16.txt  # nvinfer 설정
│   └── coco_labels.txt                   # 80개 COCO 클래스
│
├── models/
│   ├── yolo11m.onnx                      # batch=4, nms=True, opset=17
│   └── yolo11m.onnx_b4_gpu_fp16.engine   # TensorRT FP16 엔진
│
└── scripts/
    ├── build_fp16_engine.sh              # ONNX 내보내기 + TensorRT 엔진 빌드
    ├── build_yolo11_parser.sh            # parser_yolo11.cpp → lib_parser_yolo11.so
    └── export_yolo11.py                  # ultralytics ONNX export
```

**실행:**
```bash
source /home/nvidia/workspace/arround_view/venv/bin/activate
cd /home/nvidia/workspace/deepstream_yolo

python3 deepstream_yolo11_4ch_gpu_fp16.py                       # 헤드리스
DISPLAY=:0 python3 deepstream_yolo11_4ch_gpu_fp16.py            # 화면 출력
python3 deepstream_yolo11_4ch_gpu_fp16.py --output out.mp4      # 영상 저장

python3 deepstream_yolo11_power_log.py [--display] [--power-interval 0.5] [--power-csv FILE]
python3 deepstream_yolo11_frame_latency.py [--display] [--latency-csv FILE]
```


## ★ 가장 중요한 기술 사항

### 1. network-type=0 필수

`parse-bbox-func-name`으로 등록한 C++ 커스텀 파서는 **`network-type=0`(Detector)일 때만 호출**된다.  
`network-type=100`(Other)으로 설정하면 파서 함수가 **완전히 무시**되고, obj_meta도 생성되지 않아 바운딩박스가 표시되지 않는다.  
실제로 이 버그로 전 채널 `obj_meta 없음`을 확인했으며 `network-type=0`으로 변경해 해결.

### 2. nvinfer config 핵심 설정 (DS 7.0, FP16 + C++ 파서)

```ini
[property]
network-mode=2              # 2=FP16 (0=FP32, 1=INT8)
network-type=0              # 반드시 0=Detector — parse-bbox-func-name 호출 조건
custom-lib-path=pipeline/lib_parser_yolo11.so
parse-bbox-func-name=NvDsInferParseYolo11
cluster-mode=4              # NMS 엔진 내장 시 추가 클러스터링 없음
maintain-aspect-ratio=1
symmetric-padding=1         # YOLO letterbox 중앙 패딩
labelfile-path=config/coco_labels.txt

[class-attrs-all]
pre-cluster-threshold=0.30
# nms-iou-threshold는 [property]가 아닌 [class-attrs-all]에만 유효 (DS7.0 함정)
```

### 3. C++ 파서 상세 (parser_yolo11.cpp)

- ONNX export: `nms=True` → 출력 텐서 shape `[300, 6]` (x1, y1, x2, y2, conf, class_id)
- letterbox 640×640 픽셀 좌표 → nvinfer가 원본 해상도로 자동 역변환
- DS7.0 `cluster-mode=4`에서 `perClassPreclusterThreshold` 벡터가 size=0일 수 있음 → bounds 체크 필수

```cpp
const int   thresh_size = detectionParams.perClassPreclusterThreshold.size();
const float thresh = (cls < thresh_size)
                     ? detectionParams.perClassPreclusterThreshold[cls]
                     : 0.30f;   // 폴백
```

- 파서 재빌드: `bash scripts/build_yolo11_parser.sh`

### 4. nvinfer 자동 처리 (labelfile-path 설정 시)

`labelfile-path`를 설정하면 nvdsosd가 Python probe 없이 클래스명 + 바운딩박스를 자동 렌더링한다.  
`obj_meta.rect_params`의 letterbox → 원본 좌표 역변환도 nvinfer가 자동 처리.

### 5. 디스플레이 제어 패턴 (파일마다 다름)

| 파일 | 방식 |
|------|------|
| `deepstream_yolo11_4ch_gpu_fp16.py` | `DISPLAY=:0` 환경변수 |
| `deepstream_yolo11_power_log.py` | `--display` / `-d` 플래그 |
| `deepstream_yolo11_frame_latency.py` | `--display` / `-d` 플래그 |

### 6. DLA 파이프라인 — nv3dsink sync=False 필수

DLA + nv3dsink `sync=True` 조합 시 X 서버 강제 종료(segfault 139) 발생.  
DLA가 GPU를 점유하는 동안 sync=True의 GPU 클럭 타이머가 충돌하는 것이 원인.  
→ DLA 사용 파이프라인에서는 반드시 `sink.set_property("sync", False)`.

---

## 측정 프로그램 구조

### deepstream_yolo11_power_log.py

- `tegrastats --interval <ms>` subprocess 실행 후 stdout 정규식 파싱
- tegrastats에서 `VDDQ_VDD2_1V8AO`, `VDD_CPU_CV`, `VDD_GPU_SOC`, `VIN_SYS_5V0` inst 값 추출
- `all_mw`는 jtop ALL inst와 같은 레일 inst 합산값
- CSV: `timestamp, elapsed_s, vddq_vdd2_1v8ao_mw, vdd_cpu_cv_mw, vdd_gpu_soc_mw, vin_sys_5v0_mw, all_mw`

### deepstream_yolo11_frame_latency.py

- nvinfer `sink` 패드 프로브에서 PGIE 배치 진입 시각 저장
- nvinfer `src` 패드 프로브에서 배치 종료 시각과 비교
- 객체 수/채널별 row는 기록하지 않고, 처리 시간 분석에 필요한 배치 단위 정보만 CSV 저장
- CSV: `batch_seq, timestamp, elapsed_s, batch_size, pgie_batch_ms`
- CSV 쓰기는 별도 백그라운드 스레드 (queue + lock으로 probe 지연 최소화)

---


- YOLO11m DLA 레이어 분배: Conv2D/BN/ReLU → DLA, Attention(MatMul/Softmax)/검출헤드 → GPU fallback

---

## 실행 환경

- **venv:** `source /home/nvidia/workspace/arround_view/venv/bin/activate` (pyds 포함)
- **OS:** Linux 5.15.136-tegra (Jetson Orin)
- **DeepStream:** 7.0
- **디스플레이:** X11, DP-1 1920×1080
