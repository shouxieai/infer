#!/bin/bash

trtexec --onnx=workspace/yolov5s.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=workspace/yolov5s.engine