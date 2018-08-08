#pragma OPENCL EXTENSION cl_khr_fp16 : enable
float sum16(float16 a){
    return
    a.s0 + a.s1 + a.s2 + a.s3 +
    a.s4 + a.s5 + a.s6 + a.s7 +
    a.s8 + a.s9 + a.sa + a.sb +
    a.sc + a.sd + a.se + a.sf ;
}
kernel void gemm_nn9W (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k;
  int wK = K/3;
  int wlda = lda/3;
  float3 B_BUF[4608/3];
  for (j = 0; j < N; ++j) {
    for(k = 0, i = 0; k < wK; k+=3){
        B_BUF[i++] = vload_half3(( j*wlda + k + 0 ), B);
        B_BUF[i++] = vload_half3(( j*wlda + k + 1 ), B);
        B_BUF[i++] = vload_half3(( j*wlda + k + 2 ), B);
    }
    for (i = 0; i < M; ++i) {
      float Cn;
      for (k = 0, Cn = C[ i + ldc*j ];k < wK; k+=3) {
        float3 Ax1= vload_half3(( i*wlda + k + 0 ), A);
        float3 Ax2= vload_half3(( i*wlda + k + 1 ), A);
        float3 Ax3= vload_half3(( i*wlda + k + 2 ), A);
        float3 Bx1= B_BUF[( k + 0 )];
        float3 Bx2= B_BUF[( k + 1 )];
        float3 Bx3= B_BUF[( k + 2 )];
        Cn+= dot(Bx1,Ax1) + dot(Bx2,Ax2) + dot(Bx3,Ax3);
      }
      C[ i + ldc*j ] = Cn;
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
  float16 B_BUF[4608/16];
  for (j = 0; j < N; ++j) {
    for(k = 0, i = 0; k < wK; ++k)
        B_BUF[i++]= vload_half16(( j*wlda + k + 0 ), B);
    for (i = 0; i < M; ++i) {
      float Cn;
      for (k = 0, Cn = C[ i + ldc*j ];k < wK; ++k) {
        float16 Ax1= vload_half16(( i*wlda + k + 0 ), A);
        float16 Bx1= B_BUF[ k ];
        float16 Cx1= Bx1 * Ax1;
        Cn+= sum16(Cx1);
      }
      C[ i + ldc*j ] = Cn;
    }
  }
}

