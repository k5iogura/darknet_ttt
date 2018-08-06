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

#include "cl_body.h"

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
    const char *k_name[2]={"gemm_nn9W","gemm_nnfW"};
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1_emu.aocx",
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
  cl_int ret,ret1,ret2,ret3;
  memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					M * K * sizeof (float), A, &ret1);
  memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
					K * N * sizeof (float), B, &ret2);
  memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
					M * N * sizeof (float), C, &ret3);
  if(ret1 != CL_SUCCESS || ret2 != CL_SUCCESS || ret3 != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateBuffer %d %d %d\n",ret1,ret2,ret3);
	exit(ret3);
  }

  if(!(K%27))
      kernel = kernels[GEMM9W];
  else
      kernel = kernels[GEMMfW];

/* Set OpenCL Kernel Parameters */
  ret|= clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);
  ret|= clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);
  ret|= clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);
  ret|= clSetKernelArg (kernel, 3, sizeof (cl_float),&ALPHA);
  ret|= clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA);
  ret|= clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);
  ret|= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB);
  ret|= clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);
  ret|= clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC);
  ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clSetKernelArg %d\n",ret);
	exit(ret);
  }
/* Execute OpenCL Kernel */
  ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
  clFinish(command_queue);
  if(ret == CL_SUCCESS){
	  ret = clReleaseMemObject (memobjA);
	  ret = clReleaseMemObject (memobjB);
	  ret = clReleaseMemObject (memobjC);
  }else{fprintf(stderr,"clEnqueueTask Error %d\n",ret);exit(-1);}
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

