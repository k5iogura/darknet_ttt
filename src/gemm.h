#ifndef GEMM_H
#define GEMM_H

#include "half.h"
void set_FPGA_init();
void unset_FPGA_init();
int  get_FPGA_init();
int gemm_fpga_init();
void set_Nonblocking_launch();
void unset_Nonblocking_launch();
int  get_Nonblocking_launch();
void wait_kernel_finish();

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void gemm2(int TA, int TB, int TC, int M, int N, int K, float ALPHA, 
                    float *A, int lda, 
                    float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

#ifdef OPENEXR
void gemm_hf(int TA, int TB, int TC, int M, int N, int K, float ALPHA, 
                    half *A, int lda, 
                    half *B, int ldb,
                    float BETA,
                    half *C, int ldc);
#endif

void gemm_nn_binary( int M, int N, int K,
                    float *A, int lda, 
                    float *B, int ldb,
                    float *C, int ldc);

void gemm_nn_sign( int M, int N, int K,
                    float *scale,
                    unsigned int *A, int lda, 
                    float *B, int ldb,
                    float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif
#endif
