#ifndef TYPES_H
#define TYPES_H

#include <nemo_types.h>

typedef enum {
	STATUS_OK,
	STATUS_ERROR
} cpu_status_t;

#ifdef CPU_SINGLE_PRECISION
typedef float fp_t;
#else
#error "Change weight_t definitions of double precision required"
typedef double fp_t;
#endif

//typedef fp_t weight_t;


#endif
