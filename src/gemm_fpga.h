#ifdef FPGA
#include "half.h"
// GEMM on FPGA 

//Initialize FPGA Platform until to get commandQueue.
extern int gemm_fpga_init () ;

//GEMM calcuration on FPGA.
extern void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) ;

//GEMM calcuration on FPGA.
extern void gemm_ntt_fpga(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) ;

//GEMM I/F for Darknet and switching GEMM Calcuration on CPU or FPGA.
extern void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc) ;

//Release FPGA Platform.
extern void gemm_fpga_finalize();

#ifdef OPENEXR
void gemm_nn_fpga_half(int M, int N, int K, float ALPHA, 
                    half *A, int lda, 
                    half *B, int ldb,
                    half *C, int ldc);
void gemm_ntt_fpga_half(int M, int N, int K, float ALPHA, 
                    half *A, int lda, 
                    half *B, int ldb,
                    half *C, int ldc);
#endif
#endif
