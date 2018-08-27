#ifndef __DEMOX_H__
#define __DEMOX_H__

// Bridge btn detect main thread and movie thread
typedef struct{
    pthread_mutex_t img_mutex;
    pthread_mutex_t det_mutex;
    struct {
        image buff;
        int w,h,n;
        int classes;
    } img;
    struct {
        float **probs;
        box *boxes;
    } det;
} Bridge;

#endif
