![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# darknet_ttt #
### What's means "_ttt"
Darknet is an open source neural network framework written in C and CUDA. Smallest YOLO model is called "tiny-yolo", but too large for my project. My project provide FPGA version YOLO with Altera-Cyclone-V-SoC.  
So, I made "tiny-yolo" small, call "tiny-tiny-tiny-yolo".  

### depend on
1. Altera-Cyclone-V-SoC DE10Nano board.  
2. Intel SDK for FPGA 18.0  
3. Linux kernel 3.18-ltsi.  
4. OpenBLAS  
5. arm-linux-gnueabihf-  

### make and test
Before making darknet_ttt, you have to make OpenBLAS,  
$ git clone https://github.com/xianyi/OpenBLAS  
$ make  
$ make install

On DE10Nano with OpenCL_BSP sdcard.img booting(this is console linux),  
$ git clone https://github.com/k5iogura/darknet_ttt  
$ cd darknet_ttt  
$ make -f Makefile.self

for test,  
$ cd ~
$ . ./init_opencl.sh  
$ export LD_LIBRARY_PATH=/home/root/opencl_arm32_rte/host/arm32/lib:/usr/local/lib/:/opt/OpenBLAS/lib/  
$ ./darknet  
usage: ./darknet <function  
$ ./darknet detect cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/dog.jpg  
***
![running console and output image](files/detect_1file.jpeg)  
***
For MP4 Video bellow,  
$ ./darknet detector demo cfg/voc.data cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/1mb.mp4

### points
1. For running tiny-yolo model on Cortex-A9, we need reducing floating point operations. So, we modify tiny-yolo model, input image size is 224x160, output feature map size is 7x5. And reduced convolutinal layers with minimum degraded  accuracy against original tiny-yolo model. 
2. For reducing traffic btw DDR and FPGA, use half float type ieee-754-2008. This is supported by gcc for Cortex-A9 by -mfp16-format=ieee -mfpu=neon-fp16 options.
3. For speed up prediction, use gemm by OpenCL optimization and BLAS CPU optimized library.
4. For speed up visibility, split Camera process and prediction process into 2threads. Camera View is infinity loop, camera view loop and prediction loop are asynchronus. By using mutex, 2loops is synchronizing. 

### network convolution layer structure


|No|Filter|size|input|output|
|----------:|----------:|----------:|----------:|
|0:conv |16 |3x3x1|224x160x3|224x160x16|
|2:conv |32 |3x3x1|112x80x16|112x80x32 |
|4:conv |128|3x3x1|28x20x32 |28x20x128 |
|6:conv |512|3x3x1|7x5x128  |7x5x512   |
|7:conv |512|3x3x1|7x5x512  |7x5x512   |
|8:conv |256|3x3x1|7x5x512  |7x5x256   |
|9:conv |512|3x3x1|7x5x256  |7x5x512   |
|10:conv|125|3x3x1|7x5x512  |7x5x125   |


no.1, 3, 5 are muxpooling to down sampling.
### Reference for original
For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
