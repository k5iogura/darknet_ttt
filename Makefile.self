#export PKG_CONFIG_PATH=/home/kenji/libs2/opencv-2.4.13-x86_linux_gcc/installed/lib/pkgconfig:/home/kenji/libs/SDL2_x86/lib/pkgconfig:/home/kenji/libs/ffmpeg_x86+x264/lib/pkgconfig
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/opt/OpenBLAS/lib/pkgconfig
GPU=0
CUDNN=0
OPENCV?=1
OPENEXR=0
DEBUG?=0
FPGA?=1
FPGA_EMU?=0
VIEW_TIME=0
FOLDBN=1
FP32=0
BLAS=1

ARCH= \
      -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH= -gencode arch=compute_52,code=compute_52

VPATH=./src/:./examples
SLIB=libdarknet.so
ALIB=libdarknet.a
EXEC=darknet
OBJDIR=./obj/

CC=gcc
CXX=g++
NVCC=nvcc 

AR=ar
ARFLAGS=rcs
OPTS=-Ofast

LDFLAGS= -lm -pthread 
COMMON = -Iinclude/ -Isrc/

ifeq ($(BLAS),1)
CFLAGS+= -DCBLAS
LDFLAGS+=-lopenblas
endif

ifeq ($(FOLDBN),1)
CFLAGS+= -DFOLDBN
endif

AOCX=gemm1.aocx
ifeq ($(FPGA_EMU), 1) 
FPGA_DEVICE=-march=emulator
CFLAGS+= -DFPGA
OBJ+=gemm_fpga.o
CFLAGS+=  $(shell aocl compile-config)
LDFLAGS+= $(shell aocl link-config)	-lacl_emulator_kernel_rt
OPENEXR=1
else
ifeq ($(FPGA),1)
FPGA_DEVICE=
CFLAGS+= -DFPGA
OBJ+=gemm_fpga.o
CFLAGS+=  $(shell aocl compile-config)
LDFLAGS+= $(shell aocl link-config)
OPENEXR=1
endif
endif

ifeq ($(OPENEXR),1)
#CC=$(CXX)
#CFLAGS+= -DOPENEXR $(shell pkg-config --cflags IlmBase)
CFLAGS+= -DOPENEXR -mfp16-format=ieee -mfpu=neon-fp16
#LDFLAGS+=$(shell pkg-config --libs   IlmBase)
endif

ifeq ($(FP32),1)
CFLAGS+= -DFP32
GEMM1_CL= ocl/gemm1_float.cl
else
GEMM1_CL= ocl/gemm1_halfxf_halfx9.cl
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -pg
endif

ifeq ($(VIEW_TIME),1)
CFLAGS+= -DGET_LAYER_TIME
endif

ifeq ($(CC),$(CXX))
CFLAGS+=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -fpermissive -Wno-unused-variable -Wno-write-strings
else
CFLAGS+=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC -Wno-unused-variable -Wno-write-strings
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
#CFLAGS+= -DOPENCV -DSDL2
CFLAGS+= -DOPENCV
LDFLAGS+= -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab -lrt -lpthread -lm -ldl
#COMMON+= `pkg-config --cflags opencv` 
#LDFLAGS+= `pkg-config --libs sdl2` 
#COMMON+= `pkg-config --cflags sdl2` 
#OBJ+=sdl_image.o
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ+=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o detection_layer.o route_layer.o box.o normalization_layer.o avgpool_layer.o layer.o local_layer.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o crnn_layer.o demo.o batchnorm_layer.o region_layer.o reorg_layer.o tree.o  lstm_layer.o netdump.o im2row.o kn2row_conv.o data_reshape.o demoX.o
EXECOBJA=captcha.o lsd.o super.o voxel.o art.o tag.o cifar.o go.o rnn.o rnn_vid.o compare.o segmenter.o regressor.o classifier.o coco.o dice.o yolo.o detector.o  writing.o nightmare.o swag.o darknet.o 
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o gemm_gpu.o
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/darknet.h

#all: obj backup results $(SLIB) $(ALIB) $(EXEC)
all: obj  results $(SLIB) $(ALIB) $(EXEC)


tags:$(DEPS) $(wildcard src/*.c*)
	ctags $(wildcard src/* examples/* include/*)

$(EXEC): $(EXECOBJ) $(ALIB)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(ALIB) -lstdc++

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

$(AOCX):$(GEMM1_CL)
	aoc $(FPGA_DEVICE) -fpc -fp-relaxed -v -report $^ -o $(@)

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB) $(EXEC) $(EXECOBJ) $(AOCX)

