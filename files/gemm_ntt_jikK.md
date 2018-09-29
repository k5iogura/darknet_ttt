### GEMM on FPGA Fabric
***

### Abstruct

GEMM is needed at Convolutional operation of DNN. Our purpose of Project is OpenCL based FPGA Accelleration for GEMM. Darknet has GEMM functions written by C-Language, but it is sample code, therefore it's too slow.  

Darknet sample GEMM function written by naive C-Language is suitable for OpenCL implementation.  

- Can make sample GEMM function OpenCL kernel discription easy.

We make OpenCL kernel discription and compile with AOC(Altera OpenCL Compiler).

### Using FP16 ieee-754-2008 binary16

ARM gcc compiler support FP16 such as half precision floating point format. As for i386 gcc support generic FP32 format without FP16. 
- FP16 multiplication fast than FP32
- Altera share main memory btn CPU and FPGA, and Big data is  transferred btn its. This is reason why FPGA is slow. We decide to employ FP16 format to be speed up.

### Naive GEMM Algorithm
GEMM behaive, 

- C = A * B

| | | |
|-|-|-|
| | |n|
| |k|B|
|m|A|C|

 here,  
C:m x n matrix  
A:m x k matrix  
B:k x n matrix  
offset of A,B,C are lda,ldb,ldc

Generic function inteface is  
- gemm(m, n, k, A, lda, B, ldb, C, ldc)

In Darknet sample GEMM function, matirx memory layouts are
- A is row-major
- B is row-major
- C is row-major

### Suitable FPGA GEMM Algorithm
- A is row-major
- B is col-major
- C is col-major

***Using im2row function instead of generic im2col function.***  
Generic im2col copies and expands image data into col-matrix(here, B).  
On our development, we use im2row function. Therefore B matrix as weights kernel matrix, A matrix is image expanded.

***Loop order is n, m, k***  
In Naive GEMM function, loop order is m, k, n. But our implemetation use n,m,k to bufferr each B column.

***Using vloadn() OpenCL vector loading macro***  
OpenCL has vector loading macro such as vload2, vload16, vload_half16.  
AXI Bus is high perfomance Bus but SoCFPGA don't use  high band width like 128bit bus.  Therefore Bus access is slow.  We use vloadn macro to performe sequential AXI data access.

### Compile
AOC has --fpc-relaxed compile option. It is usefull to be speedup FPGA Fabirc.  Alia10 has more strong opetimization options for DNN, Cyclone familly is not supported. So sad.
