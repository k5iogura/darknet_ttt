#include "common_types.h"
#ifndef _IM2ROW_H_
#define _IM2ROW_H_
//typedef struct { int n; int c; int w; int h; } TensorDim;
void CppConvnetIm2Row(float* stacked, const float* data, int numPatchesX,
                  int numPatchesY, int numRows, const TensorDim input_dim,
                  const TensorDim filter_dim,
                  int stride, int pad);
#endif
