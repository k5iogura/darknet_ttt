#define __USE_MINGW_ANSI_STDIO 1
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef FPGA
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x500000)

#ifdef FPGA
#include "cl_body.h"
#endif

#ifdef OPENEXR
#include <OpenEXR/half.h>
#endif

//static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue command_queue;
static cl_mem memobjA = NULL;
static cl_mem memobjB = NULL;
static cl_mem memobjC = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_kernel kernels[MAX_ENV];
#define GEMM9W 0
#define GEMMfW 1
//static cl_platform_id platform_id = NULL;

int gemm_fpga_init () {
    char *emulator_flag1 = getenv("CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA");
    char *emulator_flag2 = getenv("CL_CONTEXT_EMULATOR_DEVICE_ALTERA");
    char *aocx="gemm_fpga.aocx";
    if(emulator_flag1 && !strncmp(emulator_flag1, "1", 1)){aocx = "gemm_emu.aocx",printf("emulator_mode1:%s\n",aocx);}
    if(emulator_flag2 && !strncmp(emulator_flag2, "1", 1)){aocx = "gemm_emu.aocx",printf("emulator_mode2:%s\n",aocx);}
    const char *k_name[2]={"gemm_nn9W","gemm_nnfW"};
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        aocx,
        2,
        k_name,
        &context, kernels, &command_queue
    );
    return 0;
}

void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) {
    cl_int ret=0;
    memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  M * K * sizeof (float), A, &ret);  checkErr(ret,"clCreateBuffer-0");
    memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  K * N * sizeof (float), B, &ret);  checkErr(ret,"clCreateBuffer-1");
    memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
					  M * N * sizeof (float), C, &ret);  checkErr(ret,"clCreateBuffer-2");

    if(!(K%27))
        kernel = kernels[GEMM9W];
    else
        kernel = kernels[GEMMfW];

/* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);               checkErr(ret,"clSetKernelArg-0");
    ret = clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-1");
    ret = clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-2");
    ret = clSetKernelArg (kernel, 3, sizeof (cl_float),&ALPHA);           checkErr(ret,"clSetKernelArg-3");
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-4");
    ret = clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-5");
    ret = clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-6");
    ret = clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-7");
    ret = clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC); checkErr(ret,"clSetKernelArg-8");
    ret = clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-9");

/* Execute OpenCL Kernel */
    ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);   checkErr(ret,"clEnqueueTask");
    clFinish(command_queue);
    ret = clReleaseMemObject (memobjA);
    ret = clReleaseMemObject (memobjB);
    ret = clReleaseMemObject (memobjC);
    return;
}

#ifdef OPENEXR
void gemm_nn_fpga_half(int M, int N, int K, half ALPHA, 
        half *A, int lda, 
        half *B, int ldb,
        half *C, int ldc) {
    cl_int ret=0;
    memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  M * K * sizeof (cl_half), A, &ret);  checkErr(ret,"clCreateBuffer-0");
    memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  K * N * sizeof (cl_half), B, &ret);  checkErr(ret,"clCreateBuffer-1");
    memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
					  M * N * sizeof (cl_half), C, &ret);  checkErr(ret,"clCreateBuffer-2");

    if(!(K%27))
        kernel = kernels[GEMM9W];
    else
        kernel = kernels[GEMMfW];

/* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);               checkErr(ret,"clSetKernelArg-0");
    ret = clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-1");
    ret = clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-2");
    ret = clSetKernelArg (kernel, 3, sizeof (cl_half),&ALPHA);            checkErr(ret,"clSetKernelArg-3");
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-4");
    ret = clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-5");
    ret = clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-6");
    ret = clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-7");
    ret = clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC); checkErr(ret,"clSetKernelArg-8");
    ret = clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-9");

/* Execute OpenCL Kernel */
    ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);   checkErr(ret,"clEnqueueTask");
    clFinish(command_queue);
    ret = clReleaseMemObject (memobjA);
    ret = clReleaseMemObject (memobjB);
    ret = clReleaseMemObject (memobjC);
    return;
}
#endif

void gemm_ntt_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) {
    cl_int ret=0;
    memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  M * K * sizeof (float), A, &ret);  checkErr(ret,"clCreateBuffer-0");
    memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					  K * N * sizeof (float), B, &ret);  checkErr(ret,"clCreateBuffer-1");
    memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
					  M * N * sizeof (float), C, &ret);  checkErr(ret,"clCreateBuffer-2");

    if(!(K%27))
        kernel = kernels[GEMM9W];
    else
        kernel = kernels[GEMMfW];

/* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);               checkErr(ret,"clSetKernelArg-0");
    ret = clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-1");
    ret = clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-2");
    ret = clSetKernelArg (kernel, 3, sizeof (cl_float),&ALPHA);           checkErr(ret,"clSetKernelArg-3");
    ret = clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-4");
    ret = clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);               checkErr(ret,"clSetKernelArg-5");
    ret = clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-6");
    ret = clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-7");
    ret = clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC); checkErr(ret,"clSetKernelArg-8");
    ret = clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);               checkErr(ret,"clSetKernelArg-9");

/* Execute OpenCL Kernel */
    ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);   checkErr(ret,"clEnqueueTask");
    clFinish(command_queue);
    ret = clReleaseMemObject (memobjA);
    ret = clReleaseMemObject (memobjB);
    ret = clReleaseMemObject (memobjC);
    return;
}

void gemm_fpga_finalize(){
  cl_int ret;
/* Finalization */
  ret = clFlush (command_queue);
  ret = clFinish (command_queue);
  ret = clReleaseKernel (kernel);
  ret = clReleaseProgram (program);
  ret = clReleaseCommandQueue (command_queue);
  ret = clReleaseContext (context);

  //free ((void*)source_str);

  if(ret==CL_SUCCESS)fprintf(stderr,"gemm fpga finalized.\n");
  return ;
}
#endif

