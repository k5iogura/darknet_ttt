// @file kn2row_conv.c
//
//  \date Created on: Sep 23, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#ifdef CBLAS
#include <cblas.h>
#endif
#include "common_types.h"
#include "data_reshape.h"
#include "utils.h"

#include "convolutional_layer.h"
//
// col_shift : +ve --> shift left overlap mat , -ve --> shift right overlap mat
// or shift left base mat and keep overlap mat as it is.
//
//
// row_shift : +ve (coeff is down the center coeff) --> shift up overlap mat ,
// -ve --> shift down overlap mat or shift up the base mat.
void MatrixShiftAdd(float *base_mat,
                     int base_no_rows, int base_no_cols,
                     float *overlap_mat,
                     int ov_no_rows, int ov_no_cols,
                     int row_shift, int col_shift) {
  if (row_shift == 0 && col_shift == 0 && (base_no_rows == ov_no_rows) &&
      (base_no_cols == ov_no_cols)) {
    // normal matrix add
#ifdef CBLAS
    cblas_saxpy(base_no_rows * base_no_cols, 1.0, overlap_mat, 1, base_mat, 1);
#endif
    return;
  }
  int rows_to_add, cols_to_add;
  int base_row_start, base_col_start;
  int ov_row_start, ov_col_start;
  // without padding case
  if (ov_no_rows > base_no_rows) {
    rows_to_add = base_no_rows;
    cols_to_add = base_no_cols;
    base_row_start = 0;
    base_col_start = 0;
    ov_row_start = row_shift < 0? -row_shift : 0;
    ov_col_start = col_shift < 0? -col_shift : 0;

  } else {
    rows_to_add = ov_no_rows - abs(row_shift);
    cols_to_add = ov_no_cols - abs(col_shift);

    ov_col_start = col_shift > 0? col_shift : 0;
    ov_row_start = row_shift > 0? row_shift : 0;
    base_row_start = row_shift < 0? -row_shift : 0;
    base_col_start = col_shift < 0? -col_shift : 0;
  }

  int r;
  for (r = 0; r < rows_to_add; ++r) {
    int base_mat_offset = (r + base_row_start) * base_no_cols + base_col_start;
    int overlap_mat_offset = (r + ov_row_start) * ov_no_cols + ov_col_start;
#ifdef CBLAS
    cblas_saxpy(cols_to_add, 1.0, overlap_mat + overlap_mat_offset, 1,
                base_mat + base_mat_offset, 1);
#endif
  }
}

/* Ker2Row convolution implementations.
 *
 * Assumptions:
 * 1. in_data is in NCHW format.
 * 2. filters are in MCKK format where M is the no of output maps.
 * 3. Stride will always be 1.
 * 4. pad will be zero or kernel_size / 2
 *
 * Output will be in NCHW format.
 */
static void convolutional_layer_kn2row(convolutional_layer l, network net)
/*bool Kn2RowConvLayer(const float *in_data, const float *filters,
                         const float *bias, TensorDim in_dim,
                         TensorDim filt_dim, int stride, int pad, int group,
                         float *output)*/ {
  // Currently we have limited support.
  float *in_data = net.input;
  float *filters = l.weights;
  TensorDim in_dim   = { 1, l.c, l.w, l.h };
  TensorDim filt_dim = { l.out_c, l.c, l.size, l.size };
  int stride = l.stride;
  int pad    = l.pad;
  int group  = 1;
  float *output = l.output;
  assert(group == 1);
  assert((pad == 0) || (pad == filt_dim.w / 2));
  assert(in_dim.n == 1);
  assert(filt_dim.h == filt_dim.w);
  assert(stride == 1);

  // Output dimensions.
  TensorDim out_dim;
  out_dim.w = (in_dim.w + (pad + pad) - filt_dim.w) / stride + 1;
  out_dim.h = (in_dim.h + (pad + pad) - filt_dim.h) / stride + 1;
  out_dim.c = filt_dim.n;
  out_dim.n = in_dim.n;

  // Re-arrange filters in the  k x k x no_out_maps x no_in_maps.
  // We can avoid this if the filters are already reshaped in this format.
  float *kkmc_filters = malloc(filt_dim.n * filt_dim.c * filt_dim.h *
                               filt_dim.w * sizeof(float));
  NCHW2HWNC(filters, filt_dim.n, filt_dim.c, filt_dim.h, filt_dim.w,
            kkmc_filters);

  // Just for convenience
  int H = in_dim.h;
  int W = in_dim.w;
  float alpha = 1.0;
  float beta = 0.0;

  // We need separate buffer because GEMM output will have width = H*W even
  // if there is no padding (pad = 0).
  float *gemm_output = malloc(out_dim.c * H * W * sizeof(float));

  // Prefill output buffer with bias if present else set to zero.
  /*if (bias) {
    int m, a, b;
    for (m = 0; m < out_dim.c; ++m) {
      for (a = 0; a < out_dim.h * out_dim.w; ++a) {
        output[m * out_dim.h * out_dim.w + a] = bias[m];
      }
      // For batch size > 1
      for (b = 1; b < out_dim.n; ++b) {
        memcpy(output + b * out_dim.c * out_dim.h * out_dim.w,
               output, out_dim.c * out_dim.h * out_dim.w * sizeof(float));
      }
    }
  } else {*/
//    memset(output, 0, out_dim.n * out_dim.c * out_dim.h * out_dim.w *
//           sizeof(float));
//  }

  int kr, kc, omap;
  for (kr = 0; kr < filt_dim.h; kr++) {
    int row_shift = kr - filt_dim.h / 2;
    for (kc = 0; kc < filt_dim.w; kc++) {
      int group_no = kr * filt_dim.w + kc;
      int col_shift = kc - filt_dim.w / 2;
      // Matrix dimensions - A -> mxk B -> kxn  C --> mxn
      int m = filt_dim.n;
      int k = filt_dim.c;
      int n = in_dim.h * in_dim.w;
      // This is just 1x1 convolution
#ifdef CBLAS
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m, n, k, alpha,
                  kkmc_filters + group_no * m * k,  //A
                  k, in_data,                       //B
                  n, beta,
                  gemm_output,                      //C
                  n);
#endif
      // Slide the resulting matrix which has contribution from one of the
      // KxK kernel coefficients and add to the output.
      for (omap = 0; omap < filt_dim.n; omap++) {
        MatrixShiftAdd(output + omap * out_dim.h * out_dim.w,
                        out_dim.h, out_dim.w,
                        gemm_output + omap * H * W,
                        H, W, row_shift, col_shift);
      }
    }
  }
  free(kkmc_filters);
  free(gemm_output);
//  return true;
}

void forward_convolutional_layer_kn2row(convolutional_layer l, network net){
    int out_h = l.out_h;
    int out_w = l.out_w;
//    double time=what_time_is_it_now();

    copy_cpu(l.outputs*l.batch, l.biased_output, 1, l.output, 1);
    convolutional_layer_kn2row(l,net);
    if(!l.batch_normalize){
        add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    }
}

