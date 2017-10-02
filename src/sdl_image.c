#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <SDL.h>
#include <darknet.h>
// 1. basic procedure for sdl_image functions.
//    sdlNamedWindow("SDL1",width, height); //Init SDL
//    while(1){
//        if(sdlWaitKey()) break;	//Polling Event instead of cvWaitKey()
//        sdlShowImage(in,width,height);	//Show Image on Window
//    }
//    sdlDestroyAllWindows();		//Close all Windows
static errors(const char* s){perror(s);assert(0);exit(-1);}
static SDL_Window   *window;
static SDL_Renderer *renderer;
static SDL_Texture  *texture;
#ifdef CHECK_DIS
static Uint32 Pxlfrmt=SDL_PIXELFORMAT_RGBA8888;

// Show Image on Window by Darknet Image Structure
int sdlShowImage(image p, unsigned int width, unsigned int height){
    SDL_PixelFormat *frmt = SDL_AllocFormat(Pxlfrmt);
    image copy;
    unsigned int r_offset, g_offset, b_offset;
    int x,y,z;
    void *pixels = NULL;
    unsigned int pitch=0;
    if(SDL_LockTexture(texture,NULL,&pixels,&pitch)<0) errors("SDL_LockTexture" );
    if(p.w != width || p.h != height)
        copy = resize_image(p,width,height);
    else
        copy = copy_image(p);
    r_offset=0; g_offset=copy.w*copy.h; b_offset=copy.w*copy.h*2;
    constrain_image(copy);
    if(copy.c == 3) rgbgr_image(copy);
    for(y=0;y<copy.h;y++){
        for(x=0;x<copy.w;x++){
            unsigned int index=y*copy.h+x;
            Uint8 d0 = 255.*copy.data[r_offset+index+0];
            Uint8 d1 = 255.*copy.data[g_offset+index+1];
            Uint8 d2 = 255.*copy.data[b_offset+index+2];
            *((Uint32*)pixels+index)=SDL_MapRGB(frmt,d2,d1,d0);
        }
    }
    free_image(copy);
    SDL_UnlockTexture(texture);
    if(SDL_RenderCopy(renderer, texture, NULL, NULL)<0) errors("SDL_RenderCopy");
    SDL_RenderPresent(renderer);
}

#else
static Uint32 Pxlfrmt=SDL_PIXELFORMAT_BGR24;
static SDL_mutex *sdlQF_mutex;
static SDL_Thread *th_id;
typedef struct {
    int win_w;
    int win_h;
} rendererThreadIFtype;
extern IplImage *cvQF_src;
static IplImage *intmQF_src=NULL;

// Show Image on Window
int sdlShowImage(IplImage *p, unsigned int width, unsigned int height){
    SDL_PixelFormat *frmt = SDL_AllocFormat(Pxlfrmt);
    image copy;
    void *pixels = NULL;
    unsigned int pitch=0;
#ifdef SINGLE_THREAD_SDL
    if(SDL_LockTexture(texture,NULL,&pixels,&pitch)<0) errors("SDL_LockTexture" );
    memcpy(pixels, p->imageData, p->width * p->height * p->nChannels);
    SDL_UnlockTexture(texture);
    if(SDL_RenderCopy(renderer, texture, NULL, NULL)<0) errors("SDL_RenderCopy");
    SDL_RenderPresent(renderer);
#else
    SDL_LockMutex(sdlQF_mutex);
    if(intmQF_src!=NULL) cvReleaseImage(&intmQF_src);
    intmQF_src = cvCloneImage(cvQF_src);
    SDL_UnlockMutex(sdlQF_mutex);
#endif
    return 0;
}
#endif

// renderer thread
static int rendererThread(void *args){
    IplImage *thQF_src=NULL;
    rendererThreadIFtype *p=args;
    void *pixels = NULL;
    unsigned int pitch=0;
    int win_w=p->win_w, win_h=p->win_h;
    int tex_w, tex_h;
    int quit_flag=0;
    SDL_Event event;
    SDL_LockMutex(sdlQF_mutex);
    SDL_UnlockMutex(sdlQF_mutex);
    renderer= SDL_CreateRenderer(window,-1,0);
    texture = SDL_CreateTexture(renderer,Pxlfrmt,SDL_TEXTUREACCESS_STREAMING, win_w, win_h);
    if(SDL_QueryTexture(texture,NULL,NULL,&tex_w,&tex_h)<0) errors("SDL_QueryTexture");
    if(win_w!=tex_w|| win_h!=tex_h) errors("SDL_CreateTexture incompleted\n");
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    while(1){
        SDL_LockMutex(sdlQF_mutex);
        if(thQF_src) cvReleaseImage(&thQF_src);
        if(intmQF_src) thQF_src = cvCloneImage(intmQF_src);
        SDL_UnlockMutex(sdlQF_mutex);
        if(!intmQF_src || !thQF_src) continue;
        if(SDL_LockTexture(texture,NULL,&pixels,&pitch)<0) errors("SDL_LockTexture" );
        memcpy(
          pixels,thQF_src->imageData,
          thQF_src->width*thQF_src->height*thQF_src->nChannels
        );
        SDL_UnlockTexture(texture);
        if(SDL_RenderCopy(renderer, texture, NULL, NULL)<0) errors("SDL_RenderCopy");
        SDL_RenderPresent(renderer);
        while(SDL_PollEvent(&event) == 1){
            switch(event.type){
            case SDL_KEYDOWN:
                quit_flag=1;
                event.type=SDL_KEYDOWN;
                SDL_PushEvent(&event);
                break;
            case SDL_QUIT: quit_flag=1; break;
            default:;
            }
        }
        if(quit_flag) break;
    }
    SDL_DestroyRenderer(renderer);
    return 0;
}
// Initialize SDL System
#define LCD_W 320
#define LCD_H 240
void sdlNamedWindow(const char *name, int win_w, int win_h){
    int tex_w, tex_h;
    char b[128];
    rendererThreadIFtype args={win_w,win_h};
    if(!name)
        sprintf(b,"SDL");
    else
        sprintf(b,"SDL-%s",name);
    if(SDL_Init(SDL_INIT_EVERYTHING)<0) errors("SDLInit\n");
    window  = SDL_CreateWindow(b,SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,LCD_W,LCD_H,SDL_WINDOW_FULLSCREEN);
#ifdef SINGLE_THREAD_SDL
    renderer= SDL_CreateRenderer(window,-1,0);
    texture = SDL_CreateTexture(renderer,Pxlfrmt,SDL_TEXTUREACCESS_STREAMING, win_w, win_h);
    if(SDL_QueryTexture(texture,NULL,NULL,&tex_w,&tex_h)<0) errors("SDL_QueryTexture");
    if(win_w!=tex_w|| win_h!=tex_h) errors("SDL_CreateTexture incompleted\n");
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
#else
    sdlQF_mutex = SDL_CreateMutex();
    SDL_LockMutex(sdlQF_mutex);
    th_id = SDL_CreateThread(rendererThread,"rendererThread",&args);
    sleep(1); //bug-fix:SDL_QueryTexture: Resource temporarily unavailable
    SDL_UnlockMutex(sdlQF_mutex);
#endif
}

// Poll Event
int sdlWaitKey(){
    SDL_Event event;
    int quit_flag=0;
    while(SDL_PollEvent(&event) == 1){
        switch(event.type){
        case SDL_QUIT: quit_flag=1; break;
        case SDL_KEYDOWN: quit_flag=1; break;
        default: break;
        }
    }
    return quit_flag;
}

// Close SDL System
void sdlDestroyAllWindows(){
#ifdef SINGLE_THREAD_SDL
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
#else
    int status;
    SDL_Event event;
    event.type=SDL_QUIT;
    SDL_PushEvent(&event);
    SDL_WaitThread(th_id,&status);
#endif
    SDL_DestroyWindow(window);
    SDL_Quit();
}

