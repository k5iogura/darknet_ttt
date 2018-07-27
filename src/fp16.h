#ifndef __EASY_HALF_H__
#define __EASY_HALF_H__
typedef unsigned short fp16;

typedef union {
    unsigned int n;
    float f;
} float_convert;

#ifdef __cplusplus
extern "C" {
#endif
    extern fp16 f2h(float f);
    extern float h2f(fp16 a);
    extern fp16 fp16_prod(fp16 a,fp16 b);
#ifdef __cplusplus
}
#endif
#endif
