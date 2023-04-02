cc        := g++
nvcc      = ${@CUDA_BIN}/nvcc

include_paths := src        \
			${@OPENCV_INCLUDE} \
			${@TENSORRT_INCLUDE} \
			${@CUDA_INCLUDE} \
 			${@CUDNN_INCLUDE}

library_paths := ${@OPENCV_LIB} \
			${@TENSORRT_LIB} \
			${@CUDA_LIB} \
 			${@CUDNN_LIB}

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer nvinfer_plugin nvonnxparser \
			cuda cublas cudart cudnn \
			stdc++ dl

cppstrict := -Wall -Werror -Wextra -Wno-deprecated-declarations -Wno-unused-parameter
custrict  := #-Werror=all-warnings
cpp_compile_flags := -std=c++11 -fPIC -g -fopenmp $(cppstrict) -O0
cu_compile_flags  := -std=c++11 $(custrict) -O0 -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

include ${@CUMK}