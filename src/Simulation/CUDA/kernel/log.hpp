#ifndef LOG_HPP
#define LOG_HPP

#include <stdio.h>
#include <stdlib.h>

//#define VERBOSE

/* Host logging */

#define LOG_FN                                                                \
    if(g_verbose){                                                            \
        fprintf(stderr, "%s\n", __func__);                                    \
    }


#define LOG_FN_ARGS(...)                                                      \
    if(g_verbose){                                                            \
        fprintf(stderr, "%s(", __func__);                                     \
        fprintf(stderr, __VA_ARGS__);                                         \
        fprintf(stderr, ")\n");                                               \
    }


#define LOG(msg, ...)                                                         \
    fprintf(stderr, "%s (%s): ", msg, __func__);                              \
    fprintf(stderr, __VA_ARGS__);                                             \
    fprintf(stderr, "\n");


#define WARNING(...) LOG("WARNING", __VA_ARGS__);
#define ERROR(...) LOG("ERROR", __VA_ARGS__); exit(-1);


/* Device logging in emulation mode */

#if defined(VERBOSE) && defined(__DEVICE_EMULATION__)

/* Log message in device emulation regardless of thread */
#define DEBUG_MSG(...) fprintf(stdout, __VA_ARGS__)

/* Log message in device emulation only for specific thread */
#define DEBUG_THREAD_MSG(tix, ...)                                            \
    if(threadIdx.x == tix) {                                                  \
        DEBUG_MSG(__VA_ARGS__);                                               \
    }

#else

#define DEBUG_MSG(msg, ...)
#define DEBUG_THREAD_MSG(msg, ...)

#endif

#endif
