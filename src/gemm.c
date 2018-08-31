#include "half.h"
#include "gemm.h"
#include "gemm_fpga.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef CBLAS
#include <cblas.h>
#endif

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

//#define gemm_nn gemm_nn_BcolM   // need using im2col_cpu2
//#define gemm_nn gemm_nn_naive
//#define gemm_nn gemm_nn_fp
//#define gemm_nn gemm_nn_hf
//#define gemm_nn gemm_nn_hf
#define gemm_nn gemm_nn_cblas
//#define gemm_nn gemm_nn_hetero
//#define gemm_nt gemm_nt_hf
#define gemm_nt gemm_nt_naive
#define FRACT 20
#define FIXFP int
#define FIXFPx2 long
void gemm_nn_fp(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    float maxx,minn;
    float maxA,minA;
    float maxC,minC;
    #pragma omp parallel for
    for(i = 0,maxx=-10000,maxA=-10000,minn=10000,minA=10000,maxC=-10000,minC=10000; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            FIXFP fxA_PART = (FIXFP)round(A_PART * pow(2,FRACT));
            maxA=(maxA>A[i*lda+k])?maxA:A[i*lda+k];
            minA=(minA<A[i*lda+k])?minA:A[i*lda+k];
            for(j = 0; j < N; ++j){
                maxx=(maxx>B[k*ldb+j])?maxx:B[k*ldb+j];
                minn=(minn<B[k*ldb+j])?minn:B[k*ldb+j];
                //C[i*ldc+j] += A_PART*B[k*ldb+j];
                FIXFP fxB = (FIXFP)(round(B[k*ldb+j] * pow(2,FRACT)));
                FIXFP fxC = ((FIXFPx2)fxA_PART * (FIXFPx2)fxB) >> FRACT;
                float Cn = fxC * pow(2,-FRACT);
                C[i*ldc+j] += Cn;
            }
        }
        for(j = 0; j < N; ++j){maxC=(maxC>C[i*ldc+j])?maxC:C[i*ldc+j]; minC=(minC<C[i*ldc+j])?minC:C[i*ldc+j];}
    }
    printf("A/B/C max/min = %f %f / %f %f / %f %f\n",maxA,minA,maxx,minn,maxC,minC);
}

#ifndef FPGA
#ifdef __cplusplus
void gemm_nt_hf_floatIF(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            for(k = 0; k < K; ++k){
                half A_PART = A[i*lda+k];
                half B_PART = B[k+ldb*j];
                C[i*ldc+j] += A_PART * B_PART; //OK
            }
        }
    }
}
void gemm_nt_hf_halfIF(int M, int N, int K, float ALPHA, 
        half *A, int lda, 
        half *B, int ldb,
        half *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            for(k = 0; k < K; ++k){
                half A_PART = A[i*lda+k];
                half B_PART = B[k+ldb*j];
                C[i*ldc+j] += A_PART * B_PART;
                //C[i*ldc+j] += A_PART * B[k+ldb*j];
            }
        }
    }
}
void gemm_nt_hf(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i;
    half *a=(half*)malloc(sizeof(half)*M*K);
    half *b=(half*)malloc(sizeof(half)*K*N);
    half *c=(half*)malloc(sizeof(half)*M*N);
    for(i=0;i<M*K;i++)a[i]=A[i];
    for(i=0;i<K*N;i++)b[i]=B[i];
    for(i=0;i<M*N;i++)c[i]=C[i];
    gemm_nt_hf_floatIF(M,N,K,ALPHA,A,lda,B,ldb,C,ldc);
    //gemm_nt_hf_halfIF(M,N,K,ALPHA,a,lda,b,ldb,c,ldc);
    //for(i=0;i<M*N;i++)C[i]=c[i];
    free(a);
    free(b);
    free(c);
}
void gemm_nn_hf_floatIF(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register half A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                half B_PART= B[k*ldb+j];
                C[i*ldc+j] += (float)(A_PART * B_PART); //OK
            }
        }
    }
}
void gemm_nn_hf_halfIF(int M, int N, int K, float ALPHA, 
        half *A, int lda, 
        half *B, int ldb,
        half *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            half A_PART = A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART * B[k*ldb+j];
            }
        }
    }
}
void gemm_nn_hf(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i;
    half *a=(half*)malloc(sizeof(half)*M*K);
    half *b=(half*)malloc(sizeof(half)*K*N);
    half *c=(half*)malloc(sizeof(half)*M*N);
    for(i=0;i<M*K;i++)a[i]=A[i];
    for(i=0;i<K*N;i++)b[i]=B[i];
    for(i=0;i<M*N;i++)c[i]=C[i];
    //gemm_nn_hf_floatIF(M,N,K,ALPHA,A,lda,B,ldb,C,ldc);
    gemm_nn_hf_halfIF(M,N,K,ALPHA,a,lda,b,ldb,c,ldc);
    for(i=0;i<M*N;i++)C[i]=c[i];
    free(a);
    free(b);
    free(c);
}
#endif
#endif

void gemm_nn_BcolM(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float Cn = C[i*ldc+j];
            for(k = 0; k < K; ++k){
                Cn += A[i*lda+k] * B[j*lda+k];
            }
            C[i*ldc+j] = Cn;
        }
    }
}

typedef struct gemm_args{
    int M,N,K;
    float *A, *B, *C;
    int lda,ldb,ldc;
    float ALPHA;
} gemm_args;

void gemm_nn_cblas(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
#ifdef CBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ALPHA, A, lda, B, ldb, 1, C, ldc);
#endif
}

void *gemm_nn_thrd(void *g_args){
#ifdef CBLAS
    gemm_args g = *(gemm_args *)g_args;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        g.M, g.N, g.K, g.ALPHA , g.A, g.lda, g.B, g.ldb, 1, g.C, g.ldc);
#endif
    return 0;
}
void gemm_nn_hetero(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    gemm_args *g_ptr = (gemm_args*)malloc(sizeof(gemm_args));
    g_ptr->M   =   M; g_ptr->N   =   N; g_ptr->K   =   K;
    g_ptr->lda = lda; g_ptr->ldb = ldb; g_ptr->ldc = ldc;
    g_ptr->A   =   A; g_ptr->B   =   B; g_ptr->C   =   C;
    g_ptr->ALPHA = ALPHA;
    pthread_t g_thrd;
    if(pthread_create(&g_thrd, 0, gemm_nn_thrd, g_ptr)) error("Thread creation failed hetero");
    pthread_join(g_thrd, 0);
    free(g_ptr);
}

void gemm_nn_naive(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nn_binary(int M, int N, int K,
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            float A_PART = A[i*lda+k];
            if(A_PART>0)
                for(j = 0; j < N; ++j)
                    C[i*ldc+j] += B[k*ldb+j];
            else
                for(j = 0; j < N; ++j)
                    C[i*ldc+j] -= B[k*ldb+j];
        }
        float A_PART = fabs(A[i*lda]);
        for(j = 0; j < N; ++j)
            C[i*ldc+j] *= A_PART;
    }
}

void gemm_nn_sign(int M, int N, int K,
        float *scale, unsigned int *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    unsigned int word_index, bit_index;
    unsigned int A_PART;
    //printf("K=%d\n",K);
    for(word_index = i = 0; i < M; ++i){
        for(bit_index = k = 0, A_PART = A[word_index]; k < K; ++k){
            if((A_PART & 0x1)==1){
                for(j = 0; j < N; ++j)
                    C[i*ldc+j] += B[k*ldb+j];
                //printf("plus  %d ",word_index);
            }else{
                for(j = 0; j < N; ++j)
                    C[i*ldc+j] -= B[k*ldb+j];
                //printf("minus %d ",word_index);
            }
            //prbin((float)k,A_PART);
            if(bit_index++==31){
                bit_index = 0;
                A_PART = A[++word_index];
            }else
                A_PART >>= 1;
        }
        if(bit_index!=0)
            word_index++;
        float SCALE = fabs(scale[i]);
        for(j = 0; j < N; ++j)
            C[i*ldc+j] *= SCALE;
        //printf("scale=%9.5f\n",SCALE);fflush(stdout);
    }
}

void gemm_nt_naive(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i+ldc*j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_ttn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_ttt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i+ldc*j] += sum;
        }
    }
}

static int FPGA_init=0;
static int NONBLOCKING_LAUNCH=0;
void set_FPGA_init(){FPGA_init=1;}
void unset_FPGA_init(){FPGA_init=0;}
int  get_FPGA_init(){return FPGA_init;}
void set_Nonblocking_launch(){NONBLOCKING_LAUNCH=1;}
void unset_Nonblocking_launch(){NONBLOCKING_LAUNCH=0;}
int  get_Nonblocking_launch(){return NONBLOCKING_LAUNCH;}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
//            C[i*ldc + j] *= BETA; //BUG :C-matrix is either of row or col-major
        }
    }
    if(!TA && !TB){
#ifdef FPGA
        if(!FPGA_init){FPGA_init=1;gemm_fpga_init();}
        gemm_nn_fpga(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#else
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#endif
    }else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_ttn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_ntt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)  // FPGA with im2row and col2row Model
{
    int i,j,k;
    for(j = 0; j < N; ++j){
        for(i = 0; i < M; ++i){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i+ldc*j] += sum;  //C col-major
        }
    }
}

void gemm_ntn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)  // FPGA with im2col_col_major Model
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;  //C row-major
        }
    }
}

void gemm2(int TA, int TB, int TC,
        int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i, j;
    if(!TC)
        for(i = 0; i < M; ++i) for(j = 0; j < N; ++j) C[i*ldc + j] *= BETA;
    else
        for(i = 0; i < M; ++i) for(j = 0; j < N; ++j) C[i + ldc*j] *= BETA;
#ifdef FPGA
    if(!FPGA_init){FPGA_init=1;gemm_fpga_init();}
#endif
                                 // A B C  R:Row-Major C:Col-Major
    if(!TA && !TB && !TC){       // R R R   0 0 0
#ifdef FPGA
        gemm_nn_fpga(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#else
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#endif
    }else if( TA && !TB && !TC)  // C R R   1 0 0
        gemm_tn (M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA &&  TB &&  TC)   // R C C   0 1 1 for FPGA with im2row and col2row Model
#ifdef FPGA
        gemm_ntt_fpga(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#else
        gemm_ntt     (M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#endif
    else if(!TA &&  TB && !TC)   // R C R   0 1 0 for FPGA with im2col_col_major Model
        gemm_ntn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if( TA &&  TB &&  TC)   // C C C   1 1 1
        gemm_ttt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if( TA &&  TB && !TC)   // C C R   1 1 0
        gemm_ttn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        error("not support TA,TB,TC");
}

#ifdef OPENEXR
void gemm_hf(int TA, int TB, int TC,
        int M, int N, int K, float ALPHA, 
        half *A, int lda, 
        half *B, int ldb,
        float BETA,
        half *C, int ldc)
{
    int i, j;
    if(!TC)
        for(i = 0; i < M; ++i) for(j = 0; j < N; ++j) C[i*ldc + j] *= BETA;
    else
        for(i = 0; i < M; ++i) for(j = 0; j < N; ++j) C[i + ldc*j] *= BETA;
#ifdef FPGA
    if(!FPGA_init){FPGA_init=1;gemm_fpga_init();}
#endif
                                 // A B C  R:Row-Major C:Col-Major
    if(!TA && !TB && !TC){       // R R R   0 0 0
#ifdef FPGA
        gemm_nn_fpga_half(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#else
//        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#endif
    }else if( TA && !TB && !TC)  // C R R   1 0 0
//        gemm_tn (M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        i=0;
    else if(!TA &&  TB &&  TC)   // R C C   0 1 1 for FPGA with im2row and col2row Model
#ifdef FPGA
        gemm_ntt_fpga_half(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
#else
        error("not support !TA,TB,TC");
#endif
    else if(!TA &&  TB && !TC)   // R C R   0 1 0 for FPGA with im2col_col_major Model
#ifdef FPGA
        gemm_nn_fpga_half(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#else
        gemm_nt_hf_halfIF(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
#endif
        //gemm_ntn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if( TA &&  TB &&  TC)   // C C C   1 1 1
//        gemm_ttt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        i=0;
    else if( TA &&  TB && !TC)   // C C R   1 1 0
//        gemm_ttn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
        i=0;
    else
        error("not support TA,TB,TC");
}
#endif

#ifdef GPU

#include <math.h>
#include "gemm_gpu.h"

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
#ifdef GEMM_GPU
    if(!TA && !TB)
        return gemm_nn_gpu(TA, TB, M, N, K, ALPHA, A_gpu,lda, B_gpu, BETA, ldb, C_gpu, ldc);
#endif
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

