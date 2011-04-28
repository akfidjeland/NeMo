#ifndef NEMO_CUDA_RCM_CU
#define NEMO_CUDA_RCM_CU


#include "kernel.cu_h"


__host__ __device__
size_t
rcm_metaIndexAddress(pidx_t partition, nidx_t neuron)
{
	return partition * MAX_PARTITION_SIZE + neuron;
}


#endif
