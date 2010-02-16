#ifndef NEMO_CUDA_TYPES
#define NEMO_CUDA_TYPES

#include <nemo_types.h>
#include <stdint.h>

typedef int32_t fix_t;
typedef float weight_t;  // on the host
typedef fix_t weight_dt; // on the device
typedef unsigned int pidx_t; // partition index 

/* On the device both address and weight data rae squeezed into 32b */
typedef uint32_t synapse_t;

#endif
