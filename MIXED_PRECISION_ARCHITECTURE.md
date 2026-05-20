# YOLOv8m 혼합 정밀도 모델 구조

## 개요

| 항목 | 내용 |
|------|------|
| 모델 | YOLOv8m |
| 입력 해상도 | 640 × 640 × 3 |
| 전체 TRT 레이어 수 | 315개 |
| 전략 | Backbone + Neck → **DLA INT8**, Detection Head → **GPU FP16** |
| 배치 크기 | 2 (DLA 코어 당) |

---

## 전체 구조 및 정밀도 분할

```
입력 이미지 [B, 3, 640, 640]
       │
       ▼
┌─────────────────────────────────────────────┐
│           Backbone  (model.0 ~ 8)           │  DLA INT8
│                                             │  246개 레이어
│  Conv → C2f × 4 → SPPF                     │
│  stride 2/4/8/16/32 다운샘플링              │
│                                             │
│  출력 feature map 3개:                      │
│    P3 [B, 192, 80, 80]  (소형 객체)         │
│    P4 [B, 384, 40, 40]  (중형 객체)         │
│    P5 [B, 576, 20, 20]  (대형 객체)         │
└─────────────────────────────────────────────┘
       │ P3  │ P4  │ P5
       ▼     ▼     ▼
┌─────────────────────────────────────────────┐
│           Neck / FPN-PAN  (model.9 ~ 21)    │  DLA INT8
│                                             │
│  Upsample + Concat + C2f (Top-Down)         │
│    P5 → upsample → + P4 → C2f              │
│    P4 → upsample → + P3 → C2f              │
│                                             │
│  Conv + Concat + C2f (Bottom-Up)            │
│    P3 → Conv → + P4 → C2f                 │
│    P4 → Conv → + P5 → C2f                 │
│                                             │
│  FPN 출력 3개:                              │
│    N3 [B, 192, 80, 80]                      │
│    N4 [B, 384, 40, 40]                      │
│    N5 [B, 576, 20, 20]                      │
└─────────────────────────────────────────────┘
       │ N3        │ N4        │ N5
       │           │           │
       ▼           ▼           ▼
┌─────────────────────────────────────────────┐
│         Detection Head  (model.22)          │  GPU FP16
│                                             │  69개 레이어
│  ┌─────────────┐  ┌─────────────┐           │
│  │  cv2 브랜치  │  │  cv3 브랜치  │           │
│  │ (bbox 회귀)  │  │ (클래스 확률)│           │
│  └─────────────┘  └─────────────┘           │
│                                             │
│  × 3 스케일 (P3 / P4 / P5 각각 독립 적용)   │
│                                             │
│  → DFL (분포 기반 bbox 디코딩)               │
│  → 좌표 최종 계산 (Slice / Add / Div)        │
│                                             │
│  출력:                                      │
│    output_boxes   [B, 4,  8400]             │
│    output_classes [B, 80, 8400]             │
└─────────────────────────────────────────────┘
```

---

## Detection Head 내부 구조 (GPU FP16, 69개 레이어)

Detection Head는 3개 스케일(P3/P4/P5)에 동일한 구조를 적용한다.

### cv2 — bbox 좌표 회귀 브랜치

```
N_scale  →  Conv(SiLU)  →  Conv(SiLU)  →  Conv  →  Reshape
                                            ↓
                                       [B, 64, H×W]   (DFL 분포, 16bin × 4)
```

| 레이어 | 연산 | 역할 |
|--------|------|------|
| cv2.X.0 | Conv + SiLU (Sigmoid × Mul) | 특징 추출 |
| cv2.X.1 | Conv + SiLU | 특징 정제 |
| cv2.X.2 | Conv | bbox 분포 출력 (64ch = 16bin × 4좌표) |
| Reshape | Shuffle | [B, 64, H, W] → [B, 64, H×W] |

### cv3 — 클래스 확률 브랜치

```
N_scale  →  Conv(SiLU)  →  Conv(SiLU)  →  Conv  →  Reshape
                                            ↓
                                       [B, 80, H×W]   (클래스 점수)
```

| 레이어 | 연산 | 역할 |
|--------|------|------|
| cv3.X.0 | Conv + SiLU | 특징 추출 |
| cv3.X.1 | Conv + SiLU | 특징 정제 |
| cv3.X.2 | Conv | 클래스 로짓 출력 (80ch = COCO 클래스 수) |
| Reshape | Shuffle | [B, 80, H, W] → [B, 80, H×W] |

### 3 스케일 통합 및 DFL 디코딩

```
P3 cv2 [B, 64, 6400]  ─┐
P4 cv2 [B, 64, 1600]  ──┤ Concat → [B, 64, 8400]
P5 cv2 [B, 64, 400]   ─┘
          │
          ▼
     DFL 디코더
     ┌─────────────────────────────────────────┐
     │ Reshape   [B, 64, 8400] → [B, 4, 16, 8400] │
     │ Transpose [B, 4, 16, 8400] → [B, 16, 4, 8400] │
     │ Softmax   16bin 확률 분포 정규화              │
     │ Conv(1×1) 가중합 → [B, 4, 8400]  (dist)     │
     └─────────────────────────────────────────┘
          │ dist [B, 4, 8400]
          ▼
     좌표 디코딩
     ┌──────────────────────────────────┐
     │ Slice  dist → lt [B, 2, 8400]    │  left-top 거리
     │        dist → rb [B, 2, 8400]    │  right-bottom 거리
     │ anchor_grid 기준으로 cx/cy/w/h   │
     │ cx = anchor_x - lt_x + rb_x / 2 │
     │ cy = anchor_y - lt_y + rb_y / 2 │
     │ w  = lt_x + rb_x                │
     │ h  = lt_y + rb_y                │
     └──────────────────────────────────┘
          │
          ▼
     output_boxes [B, 4, 8400]   (cx, cy, w, h  — 0~640 범위)

P3 cv3 [B, 80, 6400]  ─┐
P4 cv3 [B, 80, 1600]  ──┤ Concat → Sigmoid → output_classes [B, 80, 8400]
P5 cv3 [B, 80, 400]   ─┘
```

---

## 왜 Detection Head만 FP16인가

### INT8 양자화의 스케일 문제

YOLOv8 출력 텐서는 두 종류의 수치 범위가 혼재한다.

| 텐서 | 범위 | INT8 표현 범위 | 스케일 인수 |
|------|------|--------------|-----------|
| bbox 좌표 | 0 ~ 640 | -128 ~ 127 | 640 / 127 ≈ **5.04** |
| 클래스 확률 | 0 ~ 1 | -128 ~ 127 | 1 / 127 ≈ **0.008** |

스케일 비율이 **640배** 차이나므로 하나의 INT8 스케일로 두 텐서를 정확히 표현할 수 없다.  
split output 구조로 텐서를 분리하여 1차 완화했지만, **detection head 연산 자체가 누적하는 양자화 오차**는 남는다.

### DFL의 정밀도 민감성

DFL은 16개 bin의 확률 분포에 가중합을 적용해 좌표를 복원한다.  
INT8에서 Softmax 출력이 저해상도로 표현되면 가중합 오차가 bbox 위치에 직접 반영된다.

### GPU FP16 적용 효과

| 구간 | 이전 (전체 INT8) | 이후 (혼합 정밀도) |
|------|----------------|-----------------|
| Backbone / Neck | DLA INT8 | DLA INT8 (동일) |
| cv2 / cv3 Conv | DLA INT8 | GPU **FP16** |
| DFL 디코딩 | DLA INT8 | GPU **FP16** |
| 좌표 계산 | DLA INT8 | GPU **FP16** |
| 엔진 크기 | 28 MB | 32 MB (+4 MB) |

---

## 데이터 흐름 요약

```
[DLA Core 0]                        [GPU]
─────────────────────────────────   ──────────────────────
입력 이미지                          ↑  DLA→GPU 자동 전송
  → Backbone (model.0~8)  INT8      │  (TRT allowGPUFallback)
  → Neck/FPN (model.9~21) INT8  ───→│
                                    │  Detection Head (model.22) FP16
                                    │    cv2 × 3 scale
                                    │    cv3 × 3 scale
                                    │    DFL decoder
                                    │    coordinate decode
                                    ↓
                              output_boxes   [B, 4,  8400]
                              output_classes [B, 80, 8400]
                                    │
                              DeepStream NvDsInferParseYolov8
                              NMS (cluster-mode=2, DBSCAN)
                                    │
                              바운딩 박스 + 클래스 레이블
```

---

## 엔진 빌드 재현

```bash
# 1단계: INT8 캘리브레이션 캐시 생성 (없을 경우)
python3 build_int8_engine.py --dla-core -1

# 2단계: 혼합 정밀도 엔진 빌드
python3 build_mixed_engine.py --dla-core 0
python3 build_mixed_engine.py --dla-core 1

# 실행
python3 deepstream_yolov8_4ch_dla.py
```
