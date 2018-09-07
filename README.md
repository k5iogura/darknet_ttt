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

### make
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
usage: ./darknet <function>
$ ./darknet detect cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/dog.jpg  
![](files/detect_1file.jpeg)  
$ ./darknet detector demo cfg/voc.data cfg/ttt5_224_160.cfg data/ttt/ttt5_224_160_final.whights data/1mb.mp4

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
