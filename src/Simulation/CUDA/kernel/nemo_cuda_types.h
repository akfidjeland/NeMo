#ifndef NEMO_CUDA_TYPES
#define NEMO_CUDA_TYPES

#include <nemo_types.h>

typedef float weight_t;
typedef unsigned int pidx_t; // partition index 

/* On the device both address and weight data rae squeezed into 32b */
typedef uint32_t synapse_t;

#endif
