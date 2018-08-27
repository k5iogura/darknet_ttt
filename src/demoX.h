#ifndef __DEMOX_H__
#define __DEMOX_H__

// Bridge btn detect main thread and movie thread
typedef struct{
    struct {
        pthread_mutex_t mutex;
        image buff;
    } img;
    struct {
        pthread_mutex_t mutex;
        int w,h,n;
        int classes;
        float **probs;
        box *boxes;
    } det;
} Bridge;

#endif
