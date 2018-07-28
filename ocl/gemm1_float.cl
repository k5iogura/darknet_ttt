#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float sum9(float3 a, float3 b, float3 c){
    return
    a.s0 + a.s1 + a.s2 +
    b.s0 + b.s1 + b.s2 +
    c.s0 + c.s1 + c.s2 ;
}
float sum16(float16 a){
    return
    a.s0 + a.s1 + a.s2 + a.s3 +
    a.s4 + a.s5 + a.s6 + a.s7 +
    a.s8 + a.s9 + a.sa + a.sb +
    a.sc + a.sd + a.se + a.sf ;
}
#define WRD (9)
kernel void gemm_nn9W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  float A_PART;
  int wK = K/WRD;
  int wlda = lda/WRD;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn = C[ i*ldc + j ];
	  for (k = 0; k < wK; ++k) {
        float3 Ax1= vload3(( i*wlda + k + 0), A);
        float3 Ax2= vload3(( i*wlda + k + 3), A);
        float3 Ax3= vload3(( i*wlda + k + 6), A);
        float3 Bx1= vload3(( j*wlda + k + 0), B);
        float3 Bx2= vload3(( j*wlda + k + 3), B);
        float3 Bx3= vload3(( j*wlda + k + 6), B);
        float3 Cx1= Bx1 * Ax1;
        float3 Cx2= Bx2 * Ax2;
        float3 Cx3= Bx3 * Ax3;
        Cn+= sum9(Cx1, Cx2, Cx3);
	  }
      C[ i*ldc + j ] = Cn;
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
  float A_PART;
  int wK = K/16;
  int wlda = lda/16;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn;
      for (k = 0, Cn = C[ i*ldc + j ];k < wK; ++k) {
        float16 Ax1= vload16(( i*wlda + k ), A);
        float16 Bx1= vload16(( j*wlda + k ), B);
        float16 Cx1= Bx1 * Ax1;
        Cn+= sum16(Cx1);
      }
      C[ i*ldc + j ] = Cn;
    }
  }
}

