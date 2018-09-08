![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# darknet_ttt #
### What's means "_ttt"
Darknet is an open source neural network framework written in C and CUDA. Smallest YOLO model is called "tiny-yolo", but too large for my project. My project provide FPGA version YOLO with Altera-Cyclone-V-SoC.  
So, I made "tiny-yolo" more small, We are called it "tiny-tiny-tiny-yolo(_ttt)".  
[ttt5_224_160.cfg](https://github.com/k5iogura/darknet_ttt/blob/master/cfg/ttt5_224_160.cfg) means "tinny tiny tiny yolo revision.5 224x160 input image model(.cfg).

### depend on
1. Altera-Cyclone-V-SoC on [Terasic DE10Nano](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&No=1046) board(Rev.C) and [BSP for Intel FPGA SDK OpenCL 16.1](https://www.terasic.com.tw/cgi-bin/page/archive.pl?Language=English&CategoryNo=205&No=1046&PartNo=4)  
2. Intel FPGA SDK for OpenCL 18.0  
3. [Linux kernel 3.18-ltsi](https://github.com/k5iogura/thinkoco-linux-socfpga)  
4. [OpenBLAS](https://github.com/xianyi/OpenBLAS)  
5. toolchain gcc (self-compiler included in sdcard.img)  

### make darknet for ARM Cortex-A9 on DE10Nano
DE10Nano can self-build by installed gcc, g++.  
Before making darknet_ttt, you have to make OpenBLAS by self-build on DE10Nano,  
$ git clone https://github.com/xianyi/OpenBLAS  
$ cd OpenBLAS  
$ make  
$ make install

Next step is that making darknet executable file.
On DE10Nano with "BSP for Intel FPGA SDK OpenCL 16.1" sdcard.img for kernel booting (this is console linux) console,  
$ git clone https://github.com/k5iogura/darknet_ttt  
$ cd darknet_ttt  
$ make -f Makefile.self  
You get darknet executable.

### for test on DE10Nano,  
Execution flow are, initialize OpenCL runtime and insmod aclsoc_drv.ko, set dynamic link library path and run darknet prediction demo using 1shot picture or 1MB.MP4 or UVC Camera.  Result is in X11 Window on X11 server display.

$ cd ~
$ . ./init_opencl.sh  
$ export DISPLAY=(x11 server IP address):0  
$ export LD_LIBRARY_PATH=/home/root/opencl_arm32_rte/host/arm32/lib:/usr/local/lib/:/opt/OpenBLAS/lib/  
$ ./darknet  
usage: ./darknet <function  
$ ./darknet detect cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/dog.jpg

***  
*result of prediction about 1shot jpeg picture.*
![running console and output image](files/detect_1file.jpeg)
*ttt5_224_160.cfg can predict only BYCYCLE without DOG ;-P)*
***

for MP4 Video bellow,  
$ ./darknet detector demo cfg/voc.data cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/1mb.mp4

finaly for UVC Camera bellow,  
$ ./darknet detector demo cfg/voc.data cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights

### training for Deep Neural Network (ttt5_224_160.cfg)
pjreddie recomends ensemble training method for YOLO.  
Ensemble Training perform good result of FP acuracy.

We use nVIDIA tesla GPGPU to train.
1. classification task by Imagenet data.
2. object detection task by VOC data.  
ttt5_224_160.cfg perform VOC2012 IoU accuracy about 50% mAP([Officially tiny-YOLO is 57.1% mAP about VOC2007+2012](https://pjreddie.com/darknet/yolov2/)).

### our points
1. For running tiny-yolo model on Cortex-A9, we need reducing floating point operations. So, we modify tiny-yolo model, input image size is 224x160, output feature map size is 7x5. And reduced convolutinal layers with minimum degraded  accuracy against original tiny-yolo model. 
2. For reducing traffic btw DDR and FPGA, use half float type ieee-754-2008. This is supported by gcc for Cortex-A9 by -mfp16-format=ieee -mfpu=neon-fp16 options.  To make darknet_ttt for intel CPU, we need "half" class of OpenEXR library and g++ compiler because gcc for intel processor does not support half floating format and OpenEXR library is written by C++ language.
3. For speed up prediction, use gemm by OpenCL optimization and BLAS CPU optimized library.
4. For speed up visibility, split Camera process and prediction process into 2threads. Camera View is infinity loop, camera view loop and prediction loop are asynchronus. By using mutex, 2loops is synchronizing only at time of send/recieve image and prediction result btn themself. 
5. We use X11 client(from OpenCV) to show result of the prediction on input image. So, We need X11 server at our demonstration. DE10Nano has HDMI output port on Board. But to use HDMI port, corresponding to IP-Module for FPGA Fabric has to be impliment in FPGA Fabric. We give up using HDMI port because DE10Nano FPGA Fabric is full by OpenCL gemm for Neural Network. 

### network convolution layer structure

|-No|-Filter|-size|-input|-output|  
|----------:|----------:|----------:|----------:|  
|0 conv |16|3x3x1|224x160x3|224x160x16|  
|2 conv |32|3x3x1|112x80x16|112x80x32|  
|4 conv |128|3x3x1|28x20x32|28x20x128|  
|6 conv |512|3x3x1|7x5x128|7x5x512|  
|7 conv |512|3x3x1|7x5x512|7x5x512|  
|8 conv |256|3x3x1|7x5x512|7x5x256|  
|9 conv |512|3x3x1|7x5x256|7x5x512|  
|10 conv|125|3x3x1|7x5x512|7x5x125|  

no.1, 3, 5 are muxpooling to down sampling.
### Reference for original
For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
