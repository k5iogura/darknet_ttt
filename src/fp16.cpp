#include <string.h>
#include "fp16.h"
#include "OpenEXR/half.h"

//
// float  : IEEE 754
// fp16   : unsigned short
//
//
extern "C" fp16 f2h(float f){
    half h = f;
    fp16 p;
    memcpy(&p,&h,2);
    return   p;
}

extern "C" float h2f(fp16 p){
    half h;
    memcpy(&h,&p,2);
    float b = h;
    return    b;
}

extern "C" fp16 fp16_prod(fp16 a, fp16 b){
    half A,B;
    memcpy(&A,&a,2);
    memcpy(&B,&b,2);
    half c = A*B ;
    fp16 p;
    memcpy(&p,&c,2);
    return    p;
}

