#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define vload_half3 vload3
#define vload_half16 vload16
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
  int wK = K/3;
  int wlda = lda/3;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn;
      for (k = 0, Cn = C[ i + ldc*j ];k < wK; k+=3) {
        float3 Ax1= vload_half3(( i*wlda + k + 0 ), A);
        float3 Ax2= vload_half3(( i*wlda + k + 1 ), A);
        float3 Ax3= vload_half3(( i*wlda + k + 2 ), A);
        float3 Bx1= vload_half3(( j*wlda + k + 0 ), B);
        float3 Bx2= vload_half3(( j*wlda + k + 1 ), B);
        float3 Bx3= vload_half3(( j*wlda + k + 2 ), B);
        Cn+= dot(Bx1,Ax1) + dot(Bx2,Ax2) + dot(Bx3,Ax3);
      }
      C[ i + ldc*j ] = Cn;
    }
  }
}

kernel void gemm_nnfW (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  int wK = K/16;
  int wlda = lda/16;
  float16 ABUF[4608/16];
  for (i = 0; i < M; ++i) {
    for (k = 0;k < wK; k++) {
      ABUF[ k ] = vload_half16(( i*wlda + k + 0 ), A);
    }
    for (j = 0; j < N; ++j) {
      float Cn;
      for (k = 0, Cn = C[ i + ldc*j ];k < wK; ++k) {
        float16 Ax1= ABUF[ k ];
        float16 Bx1= vload_half16(( j*wlda + k + 0 ), B);
        float16 Cx1= Bx1 * Ax1;
        Cn+= sum16(Cx1);
      }
      C[ i + ldc*j ] = Cn;
    }
  }
}

