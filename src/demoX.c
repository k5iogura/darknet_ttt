#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demoX.h"
#include <sys/time.h>

#define DEMO 1

//#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static network net2;
static float **probs2;
static box *boxes2;
static float **predictions2;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static float **probsX;
static box    *boxesX;
static image   buffX;

static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *avg;
static double demo_time;

static Bridge bridge;

static double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

static void bridge_img2local(Bridge bridge, image buff){
    int i;
    pthread_mutex_lock(&bridge.img.mutex);
    for(i=0;i<buff.w*buff.h*buff.c;i++) buff.data[i] = bridge.img.buff.data[i];
    pthread_mutex_unlock(&bridge.img.mutex);
}

void local2bridge_img(image *buff_letter, Bridge bridge){
    int i;
    pthread_mutex_lock(&bridge.img.mutex);
    for(i=0;i<bridge.img.buff.w*bridge.img.buff.h*bridge.img.buff.c;i++)
        bridge.img.buff.data[i] = buff_letter[(buff_index + 2)%3].data[i];
    pthread_mutex_unlock(&bridge.img.mutex);
}

static void local2bridge_det(box *boxes, float **probs, Bridge bridge){
    int i,j;
    pthread_mutex_lock(&bridge.det.mutex);
        for(i=0;i<bridge.det.w*bridge.det.h*bridge.det.n;i++){
            bridge.det.boxes[i] = boxes[i];
            for(j=0;j<bridge.det.classes+1;j++)
                bridge.det.probs[i][j] = probs[i][j];
        }
    pthread_mutex_unlock(&bridge.det.mutex);
}

void bridge_det2local(Bridge bridge, box *boxes, float **probs){
    int i,j;
    pthread_mutex_lock(&bridge.det.mutex);
    for(i=0;i<bridge.det.w*bridge.det.h*bridge.det.n;i++){
        boxes[i] = bridge.det.boxes[i];
        for(j=0;j<bridge.det.classes+1;j++) probs[i][j] = bridge.det.probs[i][j];
    }
    pthread_mutex_unlock(&bridge.det.mutex);
}

static void *detect_in_threadX(void *ptr)
{
    int i,j;
    running = 1;
    float nms = .4;

    layer l = net.layers[net.n-1];
    bridge_img2local(bridge, buffX);
    float *X = buffX.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probsX, boxesX, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buffX.w, buffX.h, net.w, net.h, demo_thresh, probsX, boxesX, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxesX, probsX, l.w*l.h*l.n, l.classes, nms);
    local2bridge_det(boxesX, probsX, bridge);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    for(i = 0; i < bridge.det.w*bridge.det.h*bridge.det.n; ++i){
        int class = max_index(probsX[i], bridge.det.classes);
        float prob = probsX[i][class];
        if(prob > demo_thresh){
            printf("%s: %.0f%%\n", demo_names[class], prob*100);
        }
    }
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

static void *fetch_in_threadX(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net.w, net.h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

static void *display_in_threadX(void *ptr)
{
    image display = buff[(buff_index+2) % 3];
    draw_detections(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes);
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(10);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

static void make_bridge(layer last_layer){
    int j;
    bridge.img.buff = copy_image(buff_letter[0]);
    bridge.det.w=last_layer.w;
    bridge.det.h=last_layer.h;
    bridge.det.n=last_layer.n;
    bridge.det.classes=last_layer.classes;

    bridge.det.boxes = (box *)   calloc(last_layer.w*last_layer.h*last_layer.n, sizeof(box));
    bridge.det.probs = (float **)calloc(last_layer.w*last_layer.h*last_layer.n, sizeof(float *));
    for(j = 0; j < last_layer.w*last_layer.h*last_layer.n; ++j)
        bridge.det.probs[j] = (float *)calloc(last_layer.classes+1, sizeof(float));
}

static void *movie_loop_in_thread(void *ptr)
{
    pthread_t fetch_thread;
    while(!demo_done){
        int i,j;
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_threadX, 0)) error("Thread creation failed");

        local2bridge_img(buff_letter, bridge);
        bridge_det2local(bridge, boxes, probs);

        fps = 1./(get_wall_time() - demo_time);
        demo_time = get_wall_time();
        pthread_join(fetch_thread, 0);
        display_in_threadX(0);

    }
    return 0;
}

void demoX(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("******************************\n");
    printf("** Darknet_ttt: DemoX v.1.0 **\n");
    printf("** AOCX :gemm_ntt_jikK.aocx **\n");
    printf("** .CL  :gemm_ntt_jikK.cl   **\n");
    printf("** ttt5_224_160.cfg model   **\n");
    printf("** fp16 on ARM-gcc          **\n");
    printf("** im2row memory placement  **\n");
    printf("** fold-batch-normalization **\n");
    printf("******************************\n");
    net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    pthread_t detect_thread;
    pthread_t fetch_thread;
    pthread_t movie_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    // Areas, buffX, boxesX, probsX are for Main thread to detect.
    make_bridge(l);
    buffX = letterbox_image(buff[0], net.w, net.h);
    boxesX= (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probsX= (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probsX[j] = (float *)calloc(l.classes+1, sizeof(float));

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 100, 100);
            //cvResizeWindow("Demo", 1352, 1013);
            //cvResizeWindow("Demo", 224, 160);
            //cvResizeWindow("Demo", 640, 480);
            cvResizeWindow("Demo", 320, 240);
            cvMoveWindow("Demo",0,0);
        }
    }

    demo_time = get_wall_time();

//                              cyclic buff_index
// fetch in buff              -> 1 -> 2 -> 0 -> 1  ...
// draw from buff on window   -> 2 -> 0 -> 1 -> 2  ...
//                                  /    /    /
//                                 /    /    /
// detect & draw in buff      -> 0 -> 1 -> 2 -> 0  ...
    if(pthread_create(&movie_thread, 0, movie_loop_in_thread, 0)) error("Thread creation failed");
    while(!demo_done){
        if(demo_done) break;
        detect_in_threadX(0);
    }
    pthread_join(movie_thread, 0);
}

//#else
//void demoX(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
//{
    //fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
//}
//#endif

