cc        := g++
nvcc      = /root/.kiwi/lib/cuda-11.8/bin/nvcc

include_paths := src        \
			/root/.kiwi/lib/opencv4.2/include \
			/root/.kiwi/lib/TensorRT-8.5.3.1-cuda11x/include \
			/root/.kiwi/lib/cuda-11.8/include \
			/root/.kiwi/lib/cudnn8.6.0.163-cuda11/lib

library_paths := /root/.kiwi/lib/opencv4.2/lib \
			/root/.kiwi/lib/TensorRT-8.5.3.1-cuda11x/lib \
			/root/.kiwi/lib/cuda-11.8/lib64 \
 			/root/.kiwi/lib/cudnn8.6.0.163-cuda11/lib

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin nvonnxparser \
			cuda cublas cudart cudnn \
			stdc++ dl

cppstrict := -Wall -Werror -Wextra -Wno-deprecated-declarations -Wno-unused-parameter
custrict  := -Werror=all-warnings
cpp_compile_flags := -std=c++11 -fPIC -g -fopenmp $(cppstrict) -O0
cu_compile_flags  := -std=c++11 $(custrict) -O0 -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

include /root/.kiwi/lib/cumk/inc