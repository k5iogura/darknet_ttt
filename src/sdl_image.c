#include <stdio.h>
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
static Uint32 Pxlfrmt=SDL_PIXELFORMAT_RGBA8888;

// Show Image on Window
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

// Initialize SDL System
void sdlNamedWindow(const char *name, int win_w, int win_h){
    int tex_w, tex_h;
    char b[128];
    if(!name)
        sprintf(b,"SDL");
    else
        sprintf(b,name);
    if(SDL_Init(SDL_INIT_EVERYTHING)<0) errors("SDLInit\n");
    window  = SDL_CreateWindow(b,SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,win_w,win_h,0);
    renderer= SDL_CreateRenderer(window,-1,0);
    texture = SDL_CreateTexture(renderer,Pxlfrmt,SDL_TEXTUREACCESS_STREAMING, win_w, win_h);
    if(SDL_QueryTexture(texture,NULL,NULL,&tex_w,&tex_h)<0) errors("SDL_QueryTexture");
    if(win_w!=tex_w|| win_h!=tex_h) errors("SDL_CreateTexture incompleted\n");
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
}

// Poll Event
int sdlWaitKey(){
    SDL_Event event;
    int quit_flag=0;
    while(SDL_PollEvent(&event) == 1){
        switch(event.type){
        case SDL_QUIT: quit_flag=1; break;
        default: break;
        }
    }
    return quit_flag;
}

// Close SDL System
void sdlDestroyAllWindows(){
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

