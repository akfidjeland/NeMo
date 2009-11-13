#ifndef TYPES_H
#define TYPES_H

typedef unsigned int bool_t;

#ifdef CPU_SINGLE_PRECISION
typedef float fp_t;
#else
typedef double fp_t;
#endif

typedef fp_t weight_t;
typedef unsigned int nidx_t;
typedef unsigned int delay_t;


#endif
