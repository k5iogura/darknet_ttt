#ifdef __arm__

#ifdef OPENEXR
#define half __fp16
#endif

#else

#ifdef __cplusplus

#include <OpenEXR/half.h>

#else

Cant compile

#endif

#endif
