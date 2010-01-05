#ifndef TYPES_H
#define TYPES_H

#include <nemo_types.h>

typedef enum {
	STATUS_OK,
	STATUS_ERROR
} status_t;

#ifdef CPU_SINGLE_PRECISION
typedef float fp_t;
#else
typedef double fp_t;
#endif

typedef fp_t weight_t;


#endif
