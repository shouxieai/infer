cc        := g++
nvcc      = ${@CUDA_BIN}/nvcc

cpp_srcs := $(shell find src -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.o)
cpp_objs := $(cpp_objs:src/%=objs/%)
cpp_mk   := $(cpp_objs:.o=.mk)

cu_srcs := $(shell find src -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cuo)
cu_objs := $(cu_objs:src/%=objs/%)
cu_mk   := $(cu_objs:.cuo=.cumk)

include_paths := src        \
			${@OPENCV_INCLUDE} \
			${@TENSORRT_INCLUDE} \
			${@CUDA_INCLUDE} \
			${@CUDNN_LIB}

library_paths := ${@OPENCV_LIB} \
			${@TENSORRT_LIB} \
			${@CUDA_LIB} \
 			${@CUDNN_LIB}

link_librarys := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs \
			nvinfer \
			cuda cublas cudart cudnn \
			stdc++ dl

empty         :=
export_path   := $(subst $(empty) $(empty),:,$(library_paths))

paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=c++11 -fPIC -m64 -g -fopenmp -w -O0
cu_compile_flags  := -std=c++11 -m64 -Xcompiler -fPIC -g -w -O0
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags 		  += $(library_paths) $(link_librarys) $(paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

pro    : workspace/pro
workspace/pro : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(link_flags)

objs/%.o : src/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@g++ -c $< -o $@ $(cpp_compile_flags)

objs/%.cuo : src/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@${nvcc} -c $< -o $@ $(cu_compile_flags)

objs/%.mk : src/%.cpp
	@echo Compile depends CXX $<
	@mkdir -p $(dir $@)
	@g++ -M $< -MF $@ -MT $(@:.mk=.o) $(cpp_compile_flags)
	
objs/%.cumk : src/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@${nvcc} -M $< -MF $@ -MT $(@:.cumk=.cuo) $(cu_compile_flags)

run : workspace/pro
	@cd workspace && ./pro

clean :
	@rm -rf objs workspace/pro

.PHONY : clean yolo run

# 导出符号，使得运行时能够链接上
export LD_LIBRARY_PATH:=$(export_path):$(LD_LIBRARY_PATH)