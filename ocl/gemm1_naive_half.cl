#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void gemm_nn9W (const int M, const int N, const int K, float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k, m;
  float A_PART;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j]+= A_PART * B[k * ldb + j];
	  }
	}
  }
}
kernel void gemm_nnfW (const int M, const int N, const int K, float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k, m;
  float A_PART;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j]+= A_PART * B[k * ldb + j];
	  }
	}
  }
}
