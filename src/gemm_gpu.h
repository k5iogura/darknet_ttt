#ifndef _GEMM_GPU_H_
#define _GEMM_GPU_H_

void gemm_nn_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif

