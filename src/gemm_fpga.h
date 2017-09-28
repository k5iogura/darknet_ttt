#ifdef FPGA
// GEMM on FPGA 

//Initialize FPGA Platform until to get commandQueue.
extern int gemm_fpga_init () ;

//GEMM calcuration on FPGA.
extern void gemm_nn_fpga(int M, int N, int K, float ALPHA, 
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

#endif
