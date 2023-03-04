# 全新的tensorrt封装，轻易继承各类任务
- 轻易实现各类任务的生产者消费者模型，并进行高性能推理
- 没有复杂的封装

# 说明
- cpm.hpp 生产者消费者模型
    - 对于直接推理的任务，通过cpm.hpp可以变为自动多batch的生产者消费者模型
- infer.hpp 对tensorRT的重新封装。接口简单
- yolo.hpp 对于yolo任务的封装。基于 infer.hpp