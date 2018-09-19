# DEMO

### Structure Demo system
***

For Demo system, we are using 2 Cyclone-V-SoC Boards DE10Nano and DE0Nano.  DE10Nano is in object detection task with FPGA, DE0Nano is in X11 Server with Cortex A9. Relation btn IPaddress are bellow,    

|DE10Nano|ether|DE0Nano|
|:-|-|-:|
|192.168.138.2|====>|192.168.137.100|
|Object Detection|view|X11 server|

We have 4-Cortex-A9 CPU and 2-FPGA Fabric. 4-Cortex-A9 are in run darknet framework, USB Camera Interface, X11 server. 2-FPGA Fabric are in GEMM kernel, altvip.  We have mergin in Fabric area and CPU Power, but feature of this Demo system is low power, so we don't use all capability.  

### Reason why 2 boards needed
DE10Nano FPGA Fabric is in 2-GEMM kernels in object detection. First one is 7xN GEMM, second one 16xN GEMM, N is variable for Convolutinal Layer structure, ex. N=3~32.  We can not combine into one kernel for speed damage.  
