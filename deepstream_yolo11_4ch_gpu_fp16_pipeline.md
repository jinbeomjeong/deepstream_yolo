# deepstream_yolo11_4ch_gpu_fp16 Pipeline

## Pipeline

```mermaid
flowchart LR
    FS0["source 0<br/>filesrc"]
    FS1["source 1<br/>filesrc"]
    FS2["source 2<br/>filesrc"]
    FS3["source 3<br/>filesrc"]

    DEMUX0["qtdemux<br/>or container demux"]
    DEMUX1["qtdemux<br/>or container demux"]
    DEMUX2["qtdemux<br/>or container demux"]
    DEMUX3["qtdemux<br/>or container demux"]

    PARSE0["h264parse / h265parse"]
    PARSE1["h264parse / h265parse"]
    PARSE2["h264parse / h265parse"]
    PARSE3["h264parse / h265parse"]

    DEC0["nvv4l2decoder<br/>NVMM output"]
    DEC1["nvv4l2decoder<br/>NVMM output"]
    DEC2["nvv4l2decoder<br/>NVMM output"]
    DEC3["nvv4l2decoder<br/>NVMM output"]

    MUX["nvstreammux<br/>batch-size = NUM_SOURCES<br/>timeout = 40000 us"]
    PGIE["nvinfer / pgie<br/>YOLO11m GPU FP16<br/>NvDsInferParseYolo11"]
    TILER["nvmultistreamtiler<br/>2 x 2 layout"]
    CONV_OSD["nvvideoconvert<br/>conv-osd"]
    OSD["nvdsosd"]
    CONV_OUT["nvvideoconvert<br/>conv-out"]
    DISPLAY["nv3dsink<br/>sync = false"]

    FS0 --> DEMUX0 --> PARSE0 --> DEC0 --> MUX
    FS1 --> DEMUX1 --> PARSE1 --> DEC1 --> MUX
    FS2 --> DEMUX2 --> PARSE2 --> DEC2 --> MUX
    FS3 --> DEMUX3 --> PARSE3 --> DEC3 --> MUX

    MUX --> PGIE

    PGIE --> TILER
    TILER --> CONV_OSD --> OSD --> CONV_OUT --> DISPLAY

    classDef cpuBound fill:#eef2ff,stroke:#4f46e5,color:#111827
    classDef nvdecBound fill:#ecfeff,stroke:#0891b2,color:#111827
    classDef gpuBound fill:#ecfdf5,stroke:#059669,color:#111827
    classDef displayBound fill:#fff7ed,stroke:#ea580c,color:#111827

    class FS0,FS1,FS2,FS3,DEMUX0,DEMUX1,DEMUX2,DEMUX3,PARSE0,PARSE1,PARSE2,PARSE3 cpuBound
    class DEC0,DEC1,DEC2,DEC3 nvdecBound
    class MUX,PGIE,TILER,CONV_OSD,OSD,CONV_OUT gpuBound
    class DISPLAY displayBound
```

Legend: blue = CPU-bound / GStreamer control, cyan = NVDEC hardware decode, green = GPU / NVMM processing, orange = display sink.

## Runtime Mode

```mermaid
flowchart TD
    START["DISPLAY=:0 python deepstream_yolo11_4ch_gpu_fp16.py"] --> DISPLAY_ENV["USE_DISPLAY = true"]
    DISPLAY_ENV --> DISPLAY_MODE["Display pipeline<br/>pgie -> tiler -> osd -> nv3dsink"]
```

## Detection Metadata Flow

```mermaid
sequenceDiagram
    participant Engine as TensorRT Engine
    participant Parser as lib_parser_yolo11.so
    participant Infer as nvinfer
    participant Meta as NvDsObjectMeta
    participant OSD as nvdsosd

    Engine->>Parser: output0 [300, 6]
    Parser->>Infer: NvDsInferParseYolo11 results
    Infer->>Meta: create bbox, class_id, confidence, label
    Meta->>OSD: object metadata
    OSD->>OSD: render boxes and labels
```
