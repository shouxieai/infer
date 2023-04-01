# New tensorrt package(YoloV5 Only)
- Easily implement producer-consumer models for various tasks and perform high-performance inference
- No complex packaging, no coupling!

### Inference flow of trt
### step1 Compile the model, e.g.
`trtexec --onnx=workspace/yolov5s.onnx --saveEngine=workspace/yolov5s.engine`

### step2: Use infer inference
```bash
$ make run
```

# Reference
- [💡Video: 1. How to use TensorRT efficiently](https://www.bilibili.com/video/BV1F24y1h7LW)
- [😁Video: 2. Feeling of using Infer](https://www.bilibili.com/video/BV1B24y137nW)
- [💕Video: 3. Instance segmentation and detection of YoloV8](https://www.bilibili.com/video/BV1SY4y1C7E2)
- [😍Video: 4. Static batch & Dynamic batch](https://www.bilibili.com/video/BV15Y41167B5)
- [🌻TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)
- [🔭KIWI: Enable AI with One Click!](https://www.shouxieai.com)
