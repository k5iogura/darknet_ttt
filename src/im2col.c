#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void im2col_cpu_col_major(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = c + channels_col * (h * width_col + w);     // column-major
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

float im2col_get_pixel_noIf(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    //if (row < 0 || col < 0 ||
    //    row >= height || col >= width) return 0;
    unsigned int r0or1 = (row >= 0 && col >= 0 && row < height && col < width);
    return (float)r0or1 * im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu_mm(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col,
     int M, int N, int K, float *A, float *C) 
{
    int c,h,w;
    int i;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    float data_col2[ 224*160 ];

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;

                int col_index;
//                col_index = (c * height_col + h) * width_col + w;
//                data_col[col_index]  = im2col_get_pixel     (data_im, height, width, channels, im_row, im_col, c_im, pad);

                col_index = h*width_col + w; // 1-row only
                //data_col2 = im2col_get_pixel_noIf(data_im, height, width, channels, im_row, im_col, c_im, pad);
                data_col2[col_index] = im2col_get_pixel_noIf(data_im, height, width, channels, im_row, im_col, c_im, pad);
                //float data_col2 = im2col_get_pixel_noIf(data_im, height, width, channels, im_row, im_col, c_im, pad);
                //for (i = 0; i < M; ++i) {
                    //float A_PART = A[i*K+c];
                    //C[ i*N + h*width_col + w ] += A_PART * data_col2;
                //}
            }
        }
        for (i = 0; i < M; ++i) {
            float A_PART = A[i*K+c];
            for (h = 0; h < height_col; ++h) {
                for (w = 0; w < width_col; ++w) {
                    C[ i*N + h*width_col + w ] += A_PART * data_col2[ h*width_col + w ];
                }
            }
        }
    }
}

