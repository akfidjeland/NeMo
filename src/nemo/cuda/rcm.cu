#ifndef NEMO_CUDA_RCM_CU
#define NEMO_CUDA_RCM_CU


#include "kernel.cu_h"
#include "rcm.cu_h"


__host__ __device__
size_t
rcm_metaIndexAddress(pidx_t partition, nidx_t neuron)
{
	return partition * MAX_PARTITION_SIZE + neuron;
}



__device__
uint
rcm_indexRowStart(rcm_index_address_t addr)
{
	return addr.x;
}


__device__
uint
rcm_indexRowLength(rcm_index_address_t addr)
{
	return addr.y;
}



/*! \return address in RCM index for a neuron in current partition */
__device__
rcm_index_address_t
rcm_indexAddress(nidx_t neuron, const rcm_dt& rcm)
{
	return rcm.meta_index[rcm_metaIndexAddress(CURRENT_PARTITION, neuron)];
}


__device__
rcm_address_t
rcm_address(uint rowStart, uint rowOffset, const rcm_dt& rcm)
{
	return rcm.index[rowStart + rowOffset];
}


__device__
rsynapse_t
rcm_data(uint warpOffset, const rcm_dt& rcm)
{
	return rcm.data[warpOffset * WARP_SIZE + threadIdx.x % WARP_SIZE];
}


#endif
