#!/bin/bash

# trtexec --onnx=workspace/yolov5s.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/yolov5s.engine

trtexec --onnx=workspace/yolov8n.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=workspace/yolov8n.transd.engine

trtexec --onnx=workspace/yolov8n-seg.b1.transd.onnx \
    --saveEngine=workspace/yolov8n-seg.b1.transd.engine