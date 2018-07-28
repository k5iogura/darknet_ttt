#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float sum16(float16 a){
    return
    a.s0 + a.s1 + a.s2 + a.s3 +
    a.s4 + a.s5 + a.s6 + a.s7 +
    a.s8 + a.s9 + a.sa + a.sb +
    a.sc + a.sd + a.se + a.sf ;
}
kernel void gemm_nn9W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  float A_PART;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn = C[ i*ldc + j ];
	  for (k = 0; k < K; ++k) {
        float  Ax1= A[ i*lda + k ];
        float  Bx1= B[ j*lda + k ];
        float  Cx1= Bx1 * Ax1;
        Cn+= Cx1;
	  }
      C[ i*ldc + j ] = Cn;
	}
  }
}

kernel void gemm_nnfW (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k;
  int wK = K/16;
  int wlda = lda/16;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn;
      for (k = 0, Cn = C[ i*ldc + j ];k < wK; k+=2) {
        float16 Ax1= vload_half16(( i*wlda + k + 0 ), A);
        float16 Ax2= vload_half16(( i*wlda + k + 1 ), A);
        float16 Bx1= vload_half16(( j*wlda + k + 0 ), B);
        float16 Bx2= vload_half16(( j*wlda + k + 1 ), B);
        float16 Cx1= Bx1 * Ax1;
        float16 Cx2= Bx2 * Ax2;
        Cn+= sum16(Cx1) + sum16(Cx2);
      }
      C[ i*ldc + j ] = Cn;
    }
  }
}

