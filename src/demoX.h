#ifndef __DEMOX_H__
#define __DEMOX_H__
typedef struct{
    pthread_mutex_t img_mutex;
    pthread_mutex_t det_mutex;
    struct {
        image buff;
        int w,h,n;
        int classes;
    } img;
    float **probs;
    box *boxes;
} Bridge;
#endif
