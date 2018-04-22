#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "gemm_gpu.h"
}

// CUDA Grid & Block structure
// func<<< gridDim, blockDim >>>(args1, arg2...);
// dim3 gridDim : 2D
// dim3 blockDim: 3D
// struct {
//   int x,y,z;
// } dim3;
//
// Grid = x * y Blocks
//  _________ 
// |         |
// |         |
//y| Blocks  |
// |         |
// |         |
// |_________|
//      x
// blockIdx.x blockIdx.y
//
// Block: x * y * z Threads
//     __________
//    /         /|
//  z/         / |
//  /         /  |
// /_________/   |
// |         |   |
// |         |   |
//y| Threads |   |
// |         |  /
// |         | /
// |_________|/
//      x
// threadIdx.x threadIdx.y threadIdx.z

__global__ void gemm_nn_kenel(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<M){
    //for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                //C[i*ldc+j] += A_PART*B[k*ldb+j];
                C[i*ldc+j] += (A[i*lda+k]>0)?B[k*ldb+j]:-B[k*ldb+j];
            }
        }
    //}
    }
}
void gemm_nn_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(M/512.));
    gemm_nn_kenel<<<dimGrid,dimBlock>>>( M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    //gemm_nn_kenel<<<1,M>>>( M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}


