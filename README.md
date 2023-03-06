# New tensorrt package, easy to integrate many tasks
- Easily implement producer-consumer models for various tasks and perform high-performance inference
- No complex packaging, no coupling!

# For the Yolo-Demo
- Currently supports Yolo series 3/4/5/x/7/8
- YoloV8-Segment is supported
- 🚀 Pre-processing about 1ms
- 🚀 Post-processing about 0.5ms
![](bus.jpg)

# Description
- cpm.hpp Producer-consumer model
    - For direct inference tasks, cpm.hpp can be turned into an automatic multi-batch producer-consumer model
- infer.hpp A repackaging of tensorRT. Simple interface
- yolo.hpp Wrapper for yolo tasks. Based on infer.hpp

### Inference flow of trt
### step1 Compile the model, e.g.
`trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.engine`

### step2: Use infer inference
```c++
model = trt::load("yolov5s.engine");
... preprocess ...

// Configure the dynamic batch size.
auto dims = model->static_dims();
dims[0] = batch;
model->set_run_dims(dims);
model->forward({input_device, output_device}, stream);

... postprocess ...
```

### step2: Use yolo inference
```c++
cv::Mat image = cv::imread("image.jpg");
auto model = yolo::load("yolov5s.engine");
auto objs = model->forward(yolo::Image(image.data, image.cols, image.rows));
// use objs to draw to image. 
```


# Use of CPM (wrapping the inference as producer-consumer)
```c++
cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
cpmi.start([]{
    return yolo::load("yolov5s.engine", yolo::Type::V5);
}, batch);

auto result_futures = cpmi.commits(images);
for(auto& fut : result_futures){
    auto objs = fut.get();
    ... process ...
}
```
# Reference
- [💡Video: 1. How to use TensorRT efficiently](https://www.bilibili.com/video/BV1F24y1h7LW)
- [😁Video: 2. Feeling of using Infer](https://www.bilibili.com/video/BV1B24y137nW)
- [💕Video: 3. Instance segmentation and detection of YoloV8](https://www.bilibili.com/video/BV1SY4y1C7E2)
- [😍Video: 4. Static batch & Dynamic batch](https://www.bilibili.com/video/BV15Y41167B5)
- [🌻TensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)
- [🔭KIWI: Enable AI with One Click!](https://www.shouxieai.com)
