#include <darknet.h>
#ifdef CHECK_DIS
extern int sdlShowImage(image p, unsigned int width, unsigned int height);
#else
extern int sdlShowImage(IplImage* p, unsigned int width, unsigned int height);
extern pthread_t sdlShowImageNewThread(IplImage *clone, unsigned int width, unsigned int height);
extern void sdlShowImageJoinThread(pthread_t thread_id);
#endif

extern void sdlNamedWindow(const char *name, int win_w, int win_h);

extern int sdlWaitKey();

extern void sdlDestroyAllWindows();

