# DeepStream YOLOv8m 4채널 2×DLA INT8 파이프라인 종합 가이드

**작성일:** 2026-04-29  
**환경:** NVIDIA Jetson Orin (GA10B) · DeepStream 7.0 · TensorRT 8.6.2 · Python 3.10  
**가상환경:** `/home/nvidia/workspace/arround_view/venv`

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [개발 환경](#2-개발-환경)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [파일 구성](#4-파일-구성)
5. [빌드 및 실행 가이드](#5-빌드-및-실행-가이드)
6. [성능 결과](#6-성능-결과)
7. [Nsight Systems 프로파일링](#7-nsight-systems-프로파일링)
8. [메모리 타입 최적화](#8-메모리-타입-최적화)
9. [문제 해결 상세 기록](#9-문제-해결-상세-기록)
10. [향후 개선 방향](#10-향후-개선-방향)

---

## 1. 프로젝트 개요

NVIDIA Jetson Orin은 2개의 DLA(Deep Learning Accelerator) 코어를 내장하여 GPU를 점유하지 않고 딥러닝 추론을 병렬로 수행할 수 있다. 본 프로젝트는 이 하드웨어 특성을 최대한 활용하여 **4채널 카메라 영상에 대해 YOLOv8m 객체 인식을 실시간으로 수행**하는 DeepStream 파이프라인을 구현한다.

**YOLOv8m 선택 배경:** anchor-free 단일 스테이지 검출기, COCO val mAP 50.2%, ~25.9M 파라미터. ONNX 내보내기 시 `nms=False`로 raw 텐서 `[B, 84, N]`을 출력하므로 DeepStream 자체 NMS(`cluster-mode=2`)와 결합하기 용이하다.

기존 단일 Mux 방식(4채널 → 단일 nvstreammux → 단일 nvinfer)은 GPU 메모리 복사와 스케줄링 병목이 발생한다. 본 프로젝트는 채널을 두 그룹으로 분리하여 각 DLA 코어에 직결하는 **2×DLA 직결 구조**를 채택해 이 병목을 제거했다.

> **YOLOv8 + DLA INT8 핵심 제약:** YOLOv8m 512×512 입력 시 앵커 수 5,376개는 DLA 처리 한계(8,192)를 하회하여 검출 헤드가 DLA INT8로 실행된다. 이때 박스 좌표(~512)와 클래스 확률(0~1)이 단일 INT8 스케일로 양자화되어 클래스 확률이 0으로 소실된다. `split_yolo_output.py`로 ONNX Concat 노드를 제거하여 해결한다.

### 주요 기능

| 기능 | 내용 |
|---|---|
| 다채널 영상 입력 | 4채널 H.264 비디오 (파일 또는 RTSP 확장 가능) |
| 객체 인식 | YOLOv8m (anchor-free, DFL head), 80-class COCO, 512×512 |
| 모델 출력 | ONNX nms=False → split → `boxes[4,N]` + `classes[80,N]` |
| 정밀도 | INT8 (IInt8MinMaxCalibrator, 스케일 분리 필수) |
| 가속기 활용 | Jetson Orin DLA Core 0 + Core 1 완전 병렬 |
| 출력 | 2×2 그리드, 바운딩박스 + 클래스명 오버레이 |
| 성능 모니터 | 채널별 실시간 FPS 측정 (2초 주기) |
| 프로파일링 | Nsight Systems NVTX 어노테이션 + tegra-accelerators 트레이스 |

---

## 2. 개발 환경

### 하드웨어

| 항목 | 사양 |
|---|---|
| 플랫폼 | NVIDIA Jetson Orin |
| GPU | NVIDIA GA10B (iGPU, CPU와 DRAM 공유) |
| DLA | 2× DLA Core (Core 0, Core 1) |
| 커널 | Linux 5.15.136-tegra |

### 소프트웨어 스택

| 항목 | 버전/경로 |
|---|---|
| DeepStream SDK | 7.0 |
| TensorRT | 8.6.2 |
| CUDA | /usr/local/cuda |
| Nsight Systems | 2024.2.2 |
| Python | 3.10 |
| GStreamer | 1.0 |
| Ultralytics | YOLOv8 |
| Python 가상환경 | `/home/nvidia/workspace/arround_view/venv` |
| DeepStream 헤더 | `/opt/nvidia/deepstream/deepstream/sources/includes` |

### 주요 Python 의존성

```
gi (GObject Introspection)   # GStreamer Python 바인딩
pyds                          # DeepStream Python 바인딩
tensorrt                      # TensorRT Python API
ctypes (libcudart.so)        # pycuda 미설치 환경에서 CUDA 메모리 조작
cv2 (OpenCV)                  # 캘리브레이션 이미지 전처리
numpy                         # 배열 처리
onnx                          # ONNX 그래프 수정
nvtx                          # Nsight Systems NVTX 어노테이션
```

---

## 3. 시스템 아키텍처

### 파이프라인 구조 (2×DLA 직결)

```
uridecodebin(ch0) ─┐
uridecodebin(ch1) ─┴→ nvstreammux_dla0(batch=2, type=4) → nvinfer_dla0(DLA Core 0) ─┐
                                                                                      funnel
uridecodebin(ch2) ─┐                                                                 │
uridecodebin(ch3) ─┴→ nvstreammux_dla1(batch=2, type=4) → nvinfer_dla1(DLA Core 1) ─┘
                                                                                      │
                                                           nvmultistreamtiler(2×2) ◄──┘
                                                                    │
                                                             nvvideoconvert
                                                                    │
                                                              nvdsosd (GPU)
                                                                    │
                                                             nvvideoconvert
                                                                    │
                                                          nv3dsink (sync=False)
```

**헤드리스 모드:** `funnel → fakesink(sync=False)`

### 기존 단일 Mux 구조 대비 차별점

| 항목 | 기존 (단일 Mux) | 본 프로젝트 (2×DLA 직결) |
|---|---|---|
| Mux 구성 | 1개 (4채널 통합) | 2개 (채널 0-1 / 채널 2-3) |
| 추가 엘리먼트 | nvstreamdemux 필요 | 불필요 (제거) |
| 추론 경로 | GPU nvinfer 1개 | DLA Core 0 + Core 1 병렬 |
| GPU 점유 | 높음 | 낮음 (DLA로 오프로드) |
| 동기화 오버헤드 | mux→demux→mux 발생 | 제거 |

### YOLOv8 출력 텐서와 INT8 DLA 스케일 충돌 문제

YOLOv8m 512×512에서 앵커 수 5,376 < DLA 한계 8,192이므로 검출 헤드가 DLA INT8로 실행된다.

```
단일 텐서 output0 [84, 5376]에 대한 INT8 스케일:
  스케일 = max(박스 좌표) / 127 = 512 / 127 ≈ 4.25

클래스 확률 0.82 → INT8 = round(0.82 / 4.25) = 0  ← 소실
클래스 확률 0.25 → INT8 = round(0.25 / 4.25) = 0  ← 소실
```

**해결:** ONNX Concat 노드를 제거하여 두 텐서를 독립 출력으로 분리

```
수정 전: dbox + cls → Concat → output0 [B, 84, 5376]  (스케일 ~4.25)
수정 후: dbox → output_boxes   [B,  4, 5376]  (스케일 ~4.25, 좌표 정밀)
         cls  → output_classes [B, 80, 5376]  (스케일 ~0.008, 확률 정밀)
```

| 모델 / 입력 | 앵커 수 | DLA 한계 | 검출 헤드 위치 | split 필요 |
|---|---|---|---|---|
| YOLOv8m 512×512 (현재) | **5,376** | 8,192 | DLA INT8 | **필요** |
| YOLOv8m 640×640 | 8,400 | 8,192 | GPU FP16 fallback | 불필요 |
| YOLO11m 640×640 | 8,400 | 8,192 | GPU FP16 fallback | 불필요 |

### DLA 레이어 배치

| 레이어 유형 | 실행 위치 | 이유 |
|---|---|---|
| Conv2D, BN, ReLU, Pooling | DLA | DLA 지원 연산 |
| C2PSA Attention (MatMul) | GPU fallback | DLA MatMul 미지원 |
| 검출 헤드 Reshape/Concat | GPU fallback | DLA는 C차원 concat만 지원 |
| DFL Softmax, Div | GPU fallback | DLA 미지원 연산 |

---

## 4. 파일 구성

### 소스 파일

| 파일 | 역할 |
|---|---|
| `deepstream_yolov8_4ch_dla.py` | 메인 파이프라인 스크립트 (NVTX 어노테이션 포함) |
| `build_int8_engine.py` | TensorRT INT8 엔진 빌더 (캘리브레이션 + trtexec 호출) |
| `export_yolov8.py` | YOLOv8m ONNX 내보내기 |
| `split_yolo_output.py` | ONNX Concat 노드 제거 (INT8 스케일 충돌 해결) |
| `parser_yolov8.cpp` / `lib_parser_yolo.so` | YOLOv8 split 출력 전용 DeepStream 커스텀 파서 |
| `profile_nsys.sh` | Nsight Systems 프로파일링 실행 스크립트 |

### 설정 및 엔진 파일

| 파일 | 역할 |
|---|---|
| `config_infer_yolov8_dla0_int8.txt` | nvinfer DLA Core 0 설정 |
| `config_infer_yolov8_dla1_int8.txt` | nvinfer DLA Core 1 설정 |
| `yolov8m_512_split.onnx` | 엔진 빌드용 ONNX (Concat 제거) |
| `yolov8m_512_b100_calib_split.onnx` | 캘리브레이션 전용 ONNX (batch=100) |
| `yolov8m_512_b2_dla0_int8.engine` | DLA Core 0 TRT INT8 엔진 |
| `yolov8m_512_b2_dla1_int8.engine` | DLA Core 1 TRT INT8 엔진 |
| `yolov8m_512_int8_calib.cache` | INT8 캘리브레이션 캐시 |
| `coco_labels.txt` | COCO 80클래스 레이블 |

### nvinfer 설정 핵심

```ini
[property]
batch-size=2
network-mode=1             # INT8
use-dla-core=0             # DLA1 설정은 1
enable-dla=1
infer-dims=3;512;512
maintain-aspect-ratio=1    # letterbox 비율 유지 (정확도 필수)
symmetric-padding=1        # 대칭 패딩
cluster-mode=2             # DBSCAN NMS
output-blob-names=output_boxes;output_classes
custom-lib-path=lib_parser_yolo.so
parse-bbox-func-name=NvDsInferParseYolov8

[class-attrs-all]
nms-iou-threshold=0.45     # DS 7.0: [class-attrs-all]에만 유효
pre-cluster-threshold=0.25
```

> **주의:** `nvbuf-memory-type`은 nvinfer config의 유효하지 않은 키 (`Unknown or legacy key` 경고). Python 코드의 nvstreammux에서만 설정.

---

## 5. 빌드 및 실행 가이드

### 전제 조건

- Jetson Orin에 DeepStream 7.0, TensorRT 8.6.2 설치 완료
- `/home/nvidia/workspace/arround_view/venv` 가상환경에 Python 의존성 설치
- `nvtx` 패키지 설치: `pip install nvtx`

### 전체 실행 순서

```bash
# 0. 가상환경 활성화
source /home/nvidia/workspace/arround_view/venv/bin/activate
cd /home/nvidia/workspace/deepstream_yolo

# ── Phase 1: ONNX 생성 (최초 1회) ──────────────────────────────
# 주의: calib ONNX를 먼저 export/rename한 뒤 엔진 ONNX를 export해야
#       Ultralytics 고정 파일명(yolov8m.onnx) 덮어쓰기를 방지할 수 있다
python3 export_yolov8.py
python3 split_yolo_output.py yolov8m_512.onnx yolov8m_512_split.onnx
python3 split_yolo_output.py yolov8m_512_b100_calib.onnx yolov8m_512_b100_calib_split.onnx

# ── Phase 2: TRT 엔진 빌드 (최초 1회, ~40-60분) ────────────────
# DLA_CORE=-1: 캘리브레이션 캐시 생성 전용 (GPU 엔진 미생성)
DLA_CORE=-1 python3 build_int8_engine.py   # → yolov8m_512_int8_calib.cache
DLA_CORE=0  python3 build_int8_engine.py   # → yolov8m_512_b2_dla0_int8.engine
DLA_CORE=1  python3 build_int8_engine.py   # → yolov8m_512_b2_dla1_int8.engine

# ── Phase 3: 커스텀 파서 컴파일 (최초 1회) ──────────────────────
g++ -o lib_parser_yolo.so parser_yolov8.cpp \
    -Wall -std=c++11 -shared -fPIC \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include

# ── Phase 4: 파이프라인 실행 ─────────────────────────────────────
python3 deepstream_yolov8_4ch_dla.py video_h264.mp4           # 헤드리스
python3 deepstream_yolov8_4ch_dla.py video_h264.mp4 --display  # 디스플레이

# ── Phase 5: 성능 프로파일링 (선택) ──────────────────────────────
sudo ./profile_nsys.sh video_h264.mp4
nsys analyze profile_dla_<timestamp>.nsys-rep
```

### 엔진 빌드 내부 동작

1. 비디오에서 캘리브레이션 프레임 추출 (OpenCV)
2. `IInt8MinMaxCalibrator`로 per-tensor INT8 스케일 계산
3. 캐시 헤더 패치: `TRT-8602-MinMaxCalibration` → `TRT-8602-EntropyCalibration2`
4. `trtexec` subprocess 호출로 DLA 엔진 직렬화 (Python segfault 우회)

### 파서 빌드 주의사항

파서 내부 텐서 식별 로직: `outputLayersInfo`에서 채널 수 == 4이면 boxes, > 4이면 classes로 구분.

### 실행 결과 확인

- 터미널에 채널별 FPS 출력 (2초 주기, 목표 ≥ 25 FPS)
- 디스플레이에 2×2 그리드 + 바운딩박스 + 클래스명 표시
- 오류 없이 EOS까지 안정 동작 → 배포 완료

---

## 6. 성능 결과

### 입력 조건

| 항목 | 값 |
|---|---|
| 채널 수 | 4채널 |
| 해상도 | 1920×1080 (Full HD) |
| 코덱 | H.264 |
| 배치 크기 (DLA 코어당) | 2 |

### YOLOv8m 모델 사양

| 항목 | 값 |
|---|---|
| 파라미터 수 | ~25.9M |
| COCO val mAP@50:95 (FP32) | ~50.2% |
| 입력 해상도 | 512×512 |
| 앵커 수 | 5,376 |
| 정밀도 | INT8 |
| Confidence Threshold | 0.25 |
| IoU Threshold | 0.45 |

### 실측 FPS (Steady-state)

| 채널 | DLA 코어 | FPS |
|---|---|---|
| ch0 | DLA Core 0 | ~27 |
| ch1 | DLA Core 0 | ~27 |
| ch2 | DLA Core 1 | ~27 |
| ch3 | DLA Core 1 | ~27 |
| **합산** | **2× DLA** | **~108 FPS** |

---

## 7. Nsight Systems 프로파일링

### 프로파일링 실행

```bash
# 기본 실행 (샘플 영상, 30초)
sudo ./profile_nsys.sh

# 특정 영상
sudo ./profile_nsys.sh video_h264.mp4

# 출력: profile_dla_<timestamp>.nsys-rep + .sqlite
```

**트레이스 옵션:**
```
--trace=cuda,nvtx,cudla,osrt,tegra-accelerators
--accelerator-trace=tegra-accelerators
--gpu-metrics-device=all
--gpu-metrics-frequency=10000
--cuda-memory-usage=true
```

> **sudo + nvtx 주의:** `profile_nsys.sh`는 venv Python 절대 경로(`/home/nvidia/workspace/arround_view/venv/bin/python3`)와 `PYTHONPATH`를 명시하여 sudo 환경에서도 `nvtx` 패키지를 인식하도록 구성되어 있다.

**분석 명령:**
```bash
nsys analyze profile_dla_<timestamp>.nsys-rep
nsys analyze profile_dla_<timestamp>.nsys-rep -r cuda_memcpy_async  # 특정 rule만
# GUI: .nsys-rep 파일을 Nsight Systems 앱으로 열기 (권장)
```

### 3회 비교 결과

| 항목 | 1차 (기본 type=0) | 2차 (type=4) | 3차 (type=1, 크래시) |
|---|---|---|---|
| Pageable Memcpy 건수 | 50건 이상 | 50건 이상 | — |
| Memcpy 최대 Duration | 149,216 ns | 143,328 ns | — |
| Memcpy 데이터 크기 | 2.074 MB × 50+ | 2.074 MB × 50+ | — |
| **StreamSync 최대** | **2,961,568 ns** | **1,672,576 ns ↓43%** | — |
| **GPU Gaps** | **3개** | **2개 (1개 제거)** | — |
| **GPU 사용률 저하 구간** | **7.1% / 24.8s** | **9.2% / 24.9s** | — |

> 3차(type=1 CUDA Pinned)는 nvbufsurftransform 미지원으로 파이프라인 크래시. 결과 무효.

### 주요 발견 사항

**① `cudaStreamSynchronize` 블로킹**
- TID 97,407 (DLA Core 0) / TID 97,476 (DLA Core 1) 가 교번하며 호스트 블로킹
- type=4 적용 후: 최대 1.67 ms → StreamSync 완화에 효과적

**② GPU Gaps (엔진 로딩)**
- 시작 시 ~4.5초 idle: DLA 엔진 2개 순차 로딩 (정상)
- 0.87초 gap: 두 DLA 파이프라인 동기화 과도기

**③ GPU 사용률**
- DLA가 추론을 담당하므로 GPU(iGPU) 사용률이 낮은 것은 정상
- GPU는 디코딩(nvdec) + 전처리/후처리(nvvideoconvert, nvdsosd)만 담당

### NVTX 어노테이션 구성

| 마커 | 색상 | 구간 |
|---|---|---|
| `pipeline_init` | 초록 | `Gst.init()` 초기화 |
| `build_dla_pairs` | 파랑 | mux/nvinfer 엘리먼트 구성 |
| `build_pipeline` | 초록 | 전체 파이프라인 조립 |
| `pipeline_state_playing` | 주황 | PLAYING 전환 (엔진 로드) |
| `pipeline_running` | 보라 | 추론 실행 전체 구간 |
| `fps_probe_dla0/1` | 노랑/시안 | 프레임 단위 probe 콜백 |

---

## 8. 메모리 타입 최적화

### nvbuf-memory-type 지원 현황 (Jetson Orin nvbufsurftransform)

| type | 이름 | 지원 여부 | 비고 |
|---|---|---|---|
| 0 | NVBUF_MEM_DEFAULT | ✅ | 기본값, SURFACE_ARRAY로 매핑 |
| 1 | NVBUF_MEM_CUDA_PINNED | ❌ 크래시 | nvbufsurftransform 미지원 |
| 2 | NVBUF_MEM_CUDA_DEVICE | ❌ 크래시 | nvbufsurftransform 미지원 |
| 3 | NVBUF_MEM_CUDA_UNIFIED | ❌ 크래시 | nvbufsurftransform 미지원 |
| **4** | **NVBUF_MEM_SURFACE_ARRAY** | **✅ 채택** | **Jetson 유일 최적 타입** |

### 최종 채택 설정

```python
# deepstream_yolov8_4ch_dla.py
mux.set_property("nvbuf-memory-type", 4)  # NVBUF_MEM_SURFACE_ARRAY
```

type=0 대비 개선:
- StreamSync 최대 43% 단축 (2.96 ms → 1.67 ms)
- GPU Gap 1개 제거 (3개 → 2개)
- GPU 사용률 저하 구간 개선 (7.1% → 9.2%)

### Pageable Memcpy 잔존 분석

**현상:** `cudaMemcpy2DAsync` Device→Pageable, 2.074 MB (=1920×1080 NV12 루마 플레인), Stream ID 33 고정

**원인:** nvstreammux 내부 nvbufsurftransform이 SURFACE_ARRAY 버퍼 배치 합산 시 CUDA 내부 스테이징 복사 발생. 외부 설정으로 제어 불가.

**nvstreammux 출력 해상도 512×512로 낮추기 검토 결과 — 채택 불가:**
- nvstreammux가 1920×1080 → 512×512 강제 스트레치 시 종횡비 1.78배 왜곡 발생
- 수평 압축률: 1920/512 = 3.75×, 수직 압축률: 1080/512 = 2.11×
- YOLOv8은 letterbox 입력으로 학습됨 → 왜곡 입력에서 mAP 현저히 저하

**수용 근거:**
- Jetson Orin은 CPU·GPU가 **동일 물리 DRAM 공유 (iGPU 구조)**
- "Device→Pageable" 복사는 실제 데이터 이동이 아닌 TLB 매핑 수준
- 오버헤드: ~100 µs/batch × 54 batch/sec ≈ **5.4 ms/sec = 파이프라인의 0.3% 미만**
- **결론: type=4 설정이 정확도와 성능을 모두 만족하는 Jetson 최적 구성**

---

## 9. 문제 해결 상세 기록

### 요약 테이블

| # | 문제 | 원인 | 해결 방법 |
|---|---|---|---|
| 1 | 탐지 결과 없음 | NMS 적용 여부 불일치 (출력 형태 불일치) | `parser_yolov8.cpp` 신규 작성, `cluster-mode=2` |
| 2 | NMS IoU 설정 무시 | DS 7.0 config 스키마 변경 | `nms-iou-threshold`를 `[class-attrs-all]`로 이동 |
| 3 | 클래스명 미표시 | Python probe 미구현 | `labelfile-path=coco_labels.txt` 설정으로 자동 렌더링 |
| 4 | DLA 엔진 저장 segfault | TRT 8.6.2 Python 바인딩 버그 | `trtexec` subprocess 호출로 우회 |
| 5 | X 서버 연결 강제 종료 | DLA + nv3dsink sync=True 클럭 충돌 | `nv3dsink.set_property("sync", False)` |
| 6 | 캘리브레이터 구현 불가 | pycuda/cupy 미설치 환경 | `ctypes`로 `libcudart.so` 직접 호출 |
| 7 | 탐지 확률 0 수렴 | KL divergence 최적화로 sigmoid 출력 클리핑 | `IInt8MinMaxCalibrator`로 교체 |
| 8 | 빌드 모드 하드코딩 | `DLA_CORE`가 소스코드에 고정 | `os.environ.get("DLA_CORE", "-1")` 환경변수화 |
| 9 | 캘리브레이션 시간 과다 (~2시간) | batch=2 소배치 반복 (~1000회/pass) | 캘리브레이션 전용 ONNX batch=100 사용 (~20회/pass) |
| 10 | ONNX 파일 덮어쓰기 버그 | Ultralytics 고정 파일명 + 잘못된 순서 | calib ONNX 먼저 export/rename, 엔진 ONNX 나중에 |
| 11 | trtexec 캐시 헤더 불일치 → CUDA crash | MinMax 캐시 vs EntropyCalibration2 기대 | `patch_calib_cache_for_trtexec()` 헤더 패치 자동화 |
| 12 | 클래스 확률 전부 0 (YOLOv8m INT8 DLA) | 박스 좌표(~512)와 확률(~1) 단일 INT8 스케일 충돌 | `split_yolo_output.py`로 Concat 제거, 독립 텐서 분리 |
| 13 | nsys `ModuleNotFoundError: nvtx` | sudo 환경에서 venv 미인식 | `profile_nsys.sh`에 venv Python 절대 경로 + `PYTHONPATH` 명시 |
| 14 | nvinfer config `nvbuf-memory-type` 경고 | nvinfer config의 유효하지 않은 키 | config 파일에서 제거, Python 코드의 nvstreammux에서만 설정 |
| 15 | type=1,2,3 크래시 | nvbufsurftransform이 SURFACE_ARRAY/DEFAULT만 지원 | type=4(SURFACE_ARRAY)로 고정 |

---

### 상세 기록

#### #1 — ONNX 파서 출력 형태 불일치

**문제:** 기존 C++ 파서가 NMS 적용 후 출력 `[300, 6]`을 기대했으나, YOLOv8m ONNX는 `nms=False` 옵션으로 export되어 원시 출력 `[84, 5376]` (cx, cy, w, h + 80 classes) 반환 → 탐지 결과 없음.

**해결:** `parser_yolov8.cpp` 작성: `[84, 5376]` 텐서를 직접 파싱하고 DeepStream NMS(`cluster-mode=2`)에 넘기는 구조로 변경.

```ini
custom-lib-path=lib_parser_yolo.so
parse-bbox-func-name=NvDsInferParseYolov8
cluster-mode=2
```

---

#### #2 — DS 7.0 `nms-iou-threshold` 위치 오류

**문제:**
```ini
[property]
nms-iou-threshold=0.45   # ← "Unknown or legacy key" 경고 후 무시됨
```

**해결:** DS 7.0부터 `[class-attrs-all]` 섹션에 작성해야 적용됨.
```ini
[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
```

---

#### #3 — Python probe 없이 클래스명 렌더링

**문제:** nvdsosd가 클래스명을 표시하지 않아 탐지 박스만 그려짐.

**해결:** nvinfer config에 `labelfile-path=coco_labels.txt` 설정 시, nvinfer가 자동으로 아래 항목을 채움:
- `obj_meta.obj_label` ← coco_labels.txt[class_id]
- `obj_meta.text_params` (display_text, 좌표, 폰트, 배경)
- `obj_meta.rect_params` (letterbox → 원본 좌표 역변환)

Python probe 코드 불필요, `nvdsosd display-text=1`(기본값)으로 렌더링.

---

#### #4 — TRT Python API DLA 엔진 직렬화 segfault

**문제:** `builder.build_serialized_network(network, config)` 호출 후 DLA 엔진 저장 시 exit code 139(segfault) 발생. TRT 8.6.2 Python 바인딩의 알려진 버그.

**해결:** DLA 빌드에 한해 C++ 바이너리 `trtexec`를 subprocess로 호출.

```python
cmd = ["/usr/src/tensorrt/bin/trtexec",
       f"--onnx={ONNX_PATH}", f"--saveEngine={engine_path}",
       "--int8", "--fp16",
       f"--useDLACore={DLA_CORE}", "--allowGPUFallback",
       f"--calib={CALIB_CACHE}", ...]
subprocess.run(cmd, check=False)
```

---

#### #5 — nv3dsink + sync=True → X 서버 연결 강제 종료

**문제:** DLA 파이프라인 실행 중 `"X connection to :0 broken (explicit kill or server shutdown)"` → segfault.  
원인: DLA가 GPU를 점유하는 동안 nv3dsink의 sync=True 모드가 GPU 클럭 타이머를 사용하려 해 충돌.

**해결:**
```python
sink.set_property("sync", False)   # DLA 파이프라인 필수
```

**시도 결과 요약:**

| 시도 | 결과 |
|---|---|
| `DISPLAY=` (빈 값) | nvbufsurface EGL 초기화 실패 |
| `DISPLAY=:0`, sync=True | X 연결 강제 종료 |
| `nveglglessink` | caps 협상 실패 |
| `kmssink` | X 서버가 DRM master 점유 |
| `DISPLAY=:0`, sync=False | **정상 동작** ✓ |

---

#### #6 — pycuda 미설치 환경에서 INT8 캘리브레이터 구현

**문제:** `pycuda`, `cupy`, `cuda-python` 미설치 → 기존 캘리브레이터 예제 코드 동작 불가.

**해결:** `ctypes`로 `libcudart.so` 직접 호출하여 GPU 메모리 할당/복사 구현.

```python
libcudart = ctypes.CDLL("libcudart.so")
libcudart.cudaMalloc.restype = ctypes.c_int
# cudaMalloc, cudaMemcpy, cudaFree 직접 호출
```

---

#### #7 — IInt8EntropyCalibrator2 사용 시 탐지 확률 0 수렴

**문제:** `IInt8EntropyCalibrator2`로 캘리브레이션 시 YOLO sigmoid 출력(고신뢰도 앵커)이 KL divergence 최적화 과정에서 클리핑됨 → 모든 클래스 확률 ~0 → 탐지 불가.

**해결:** `IInt8MinMaxCalibrator` 사용. 실제 최대값을 그대로 보존하여 클래스 스코어 범위 정확히 캘리브레이션.

```python
class YOLOInt8Calibrator(trt.IInt8MinMaxCalibrator):
    ...
```

---

#### #8 — DLA_CORE 하드코딩 → 환경변수 전환

**문제:** `DLA_CORE = -1`이 스크립트에 하드코딩되어 있어 GPU/DLA0/DLA1 빌드마다 파일을 직접 수정해야 함.

**해결:** 환경변수로 주입 가능하도록 변경.

```python
DLA_CORE = int(os.environ.get("DLA_CORE", "-1"))
```

```bash
DLA_CORE=-1 python3 build_int8_engine.py   # 캘리브레이션 캐시 생성
DLA_CORE=0  python3 build_int8_engine.py   # DLA Core 0 엔진
DLA_CORE=1  python3 build_int8_engine.py   # DLA Core 1 엔진
```

---

#### #9 — 캘리브레이션 시간 과다 (batch=2, ~2시간)

**문제:** 엔진 배치 크기(batch=2)로 캘리브레이션 시:
```
2000장 / 2 = 1000회/pass × 3~4 pass ≈ 3000~4000회 × ~2초 ≈ 1~2시간
```

**해결:** 캘리브레이션 전용 ONNX(batch=100)를 별도 사용.

```
캘리브레이션: yolov8m_512_b100_calib_split.onnx (batch=100) → 2000/100 = 20회/pass
엔진 빌드   : yolov8m_512_split.onnx (batch=2) + 생성된 캐시 사용
```

---

#### #10 — ONNX export 순서 오류 → 파일 덮어쓰기 버그

**문제:** Ultralytics는 항상 `yolov8m.onnx`로 저장. export 순서가 잘못되면:
1. batch=2 export → `yolov8m.onnx` ✓
2. batch=100 export → `yolov8m.onnx` 덮어씀 ❌
3. rename → `yolov8m_512_b100_calib.onnx`
4. 결과: `yolov8m.onnx` (batch=2) 없음 → `FileNotFoundError`

**해결:** export 순서를 역전: calib ONNX 먼저 export 및 rename, 엔진 ONNX 나중에 export.

```python
# 1. calib ONNX 먼저 export → rename
model.export(batch=CALIB_BATCH_SIZE, ...)
os.rename("yolov8m.onnx", CALIB_ONNX)

# 2. 엔진 ONNX export (yolov8m.onnx로 유지)
model.export(batch=BATCH_SIZE, ...)
```

---

#### #11 — trtexec 캘리브레이션 캐시 헤더 불일치 → CUDA crash

**문제:** Python `IInt8MinMaxCalibrator`가 생성한 캐시 헤더:
```
TRT-8602-MinMaxCalibration
```
trtexec가 기대하는 헤더:
```
TRT-8602-EntropyCalibration2
```
헤더 불일치 → trtexec가 캐시를 버리고 빈 데이터로 재캘리브레이션 시도  
→ `Assertion context->executeV2(&bindings[0]) failed` + CUDA illegal memory access crash

**해결:** 캐시 생성 직후 헤더 첫 줄만 패치 (per-tensor 수치는 그대로 유지하므로 탐지 정확도 영향 없음).

```python
def patch_calib_cache_for_trtexec(cache_file):
    with open(cache_file, "r") as f:
        lines = f.readlines()
    if lines and "MinMaxCalibration" in lines[0]:
        lines[0] = lines[0].replace("MinMaxCalibration", "EntropyCalibration2")
        with open(cache_file, "w") as f:
            f.writelines(lines)
```

`generate_calib_cache()` 완료 후 자동 호출.

---

#### #12 — YOLOv8m INT8 DLA에서 클래스 확률 전부 0 소실

**문제:** YOLOv8m 512×512 INT8 DLA 엔진 실행 시 바운딩 박스 미출력. 파서 디버그: `max_conf=0.0000`

**원인:**

| 채널 | 내용 | 값 범위 |
|---|---|---|
| [0:4] | cx, cy, w, h (박스 좌표) | **[0, 512]** |
| [4:84] | 클래스 확률 (sigmoid) | **[0, 1]** |

```
scale = 512 / 127 ≈ 4.25
클래스 확률 0.82 → INT8 = round(0.82 / 4.25) = 0 → 탐지 불가
```

YOLO11m 640×640에서 미발생 이유: 앵커 수 8,400 > DLA 최대 8,192 → 검출 헤드 전체가 GPU FP16으로 실행.  
YOLOv8m 512×512: 앵커 수 5,376 < 8,192 → 검출 헤드가 DLA INT8로 실행 → 스케일 충돌 발생.

**해결:** ONNX의 최종 Concat 노드를 제거하여 텐서를 별도 출력으로 분리.

```
수정 전: dbox + cls → Concat → output0 [84, N]  (단일 스케일 4.25)
수정 후: dbox → output_boxes   [4,  N]  (스케일 ~4.25)
         cls  → output_classes [80, N]  (스케일 ~0.008)
```

수정 후 캘리브레이션 스케일:
- `output_boxes: 4.271` → 좌표 [0,512] 정밀도 유지
- `output_classes: 0.00784` → 확률 0.82 → INT8=105 → 복원 0.823 ✓

```bash
python3 split_yolo_output.py yolov8m_512.onnx yolov8m_512_split.onnx
python3 split_yolo_output.py yolov8m_512_b100_calib.onnx yolov8m_512_b100_calib_split.onnx
```

> **주의:** Slice로 output0을 나누는 방식은 **효과 없음** — output0이 이미 INT8로 손상된 후 Slice 적용. 반드시 Concat **이전** 텐서를 직접 출력으로 사용해야 함.

---

#### #13 — sudo 환경에서 nsys ModuleNotFoundError: nvtx

**문제:** `sudo ./profile_nsys.sh` 실행 시 `ModuleNotFoundError: No module named 'nvtx'` — sudo 환경에서 venv Python 경로 미인식.

**해결:** `profile_nsys.sh`에 venv Python 절대 경로와 `PYTHONPATH` 명시.

```bash
# profile_nsys.sh 내부
PYTHON=/home/nvidia/workspace/arround_view/venv/bin/python3
PYTHONPATH=/home/nvidia/workspace/arround_view/venv/lib/python3.10/site-packages
nsys profile ... $PYTHON deepstream_yolov8_4ch_dla.py ...
```

---

#### #14 — nvinfer config에서 nvbuf-memory-type 경고

**문제:** nvinfer config에 `nvbuf-memory-type` 키를 작성하면 `Unknown or legacy key` 경고가 출력되고 무시됨.

**해결:** nvinfer config 파일에서 해당 키 제거, Python 코드에서 nvstreammux 엘리먼트에만 설정.

```python
mux.set_property("nvbuf-memory-type", 4)
```

---

#### #15 — nvbuf-memory-type 1·2·3 → nvbufsurftransform 크래시

**문제:** Jetson Orin에서 `nvbuf-memory-type`을 1(CUDA Pinned), 2(CUDA Device), 3(CUDA Unified)로 설정하면 nvbufsurftransform 단계에서 파이프라인 크래시.

**해결:** Jetson에서 nvbufsurftransform은 `NVBUF_MEM_DEFAULT(0)`와 `NVBUF_MEM_SURFACE_ARRAY(4)`만 지원. type=4로 고정.

---

## 10. 향후 개선 방향

### 모델 업그레이드 (YOLOv8 제약 해소)

| 후보 모델 | 앵커 수 | 검출 헤드 위치 | split 필요 | mAP@50:95 |
|---|---|---|---|---|
| YOLOv8m 512×512 (현재) | 5,376 | DLA INT8 | **필요** | ~50.2% |
| YOLOv8m 640×640 | 8,400 | GPU FP16 | 불필요 | ~50.2% |
| YOLO11m 640×640 | 8,400 | GPU FP16 | 불필요 | ~51.5% |

**권장:** YOLO11m 640×640 전환 시 `split_yolo_output.py`, `output-blob-names` 분리 설정, 파서 분기 로직을 모두 제거하고 단일 출력 파서로 단순화 가능.

### 입력 소스 확장

- **RTSP 카메라:** `uridecodebin` URI를 `rtsp://` 형식으로 변경만으로 즉시 적용
- **USB 카메라:** `v4l2src` + `nvvideoconvert` 연결
- **동적 소스:** DeepStream Dynamic Source 패턴 적용

### 채널 수 확장

- **8채널:** DLA 코어당 batch=4로 확장 (DLA 메모리 한계 검토 필요)
- **다중 Orin 클러스터:** 노드 간 RTSP 스트리밍으로 수평 확장

### 파이프라인 고도화

- **후처리 커스터마이징:** 특정 클래스 필터링, ROI 기반 경보
- **메타데이터 저장:** Kafka/Redis 연동으로 탐지 결과 외부 전송
- **빌드 자동화:** Makefile 또는 shell script로 Phase 1~3을 단일 명령으로 통합
- **Docker 컨테이너화:** 의존성 패키징으로 배포 용이성 향상
