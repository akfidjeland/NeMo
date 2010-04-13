#ifndef NEMO_CUDA_TYPES
#define NEMO_CUDA_TYPES

#include <nemo_types.h>
#include <stdint.h>

typedef int32_t fix_t;
typedef fix_t weight_dt; // on the device
typedef unsigned int pidx_t; // partition index 

/* On the device both address and weight data are squeezed into 32b */
//! \todo use a union type here?
typedef uint32_t synapse_t;

/* Type for storing (within-partition) neuron indices on the device. We could
 * use uint16_t here to save some shared memory, in exchange for slightly
 * poorer shared memory access patterns */
typedef uint32_t nidx_dt;

#endif
