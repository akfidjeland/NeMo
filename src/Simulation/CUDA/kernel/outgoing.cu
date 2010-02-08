#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

#include "outgoing.cu_h"

__constant__ size_t c_outgoingPitch; // word pitch



__host__
outgoing_t
make_outgoing(pidx_t partition, delay_t delay, uint warp,
		uint warpOffset,
		uint32_t warpTargetBits)
{
	//! \todo could share pointer packing with dispatchTable code
	assert(partition < MAX_PARTITION_COUNT);
	assert(delay < MAX_DELAY);
	assert(warp < MAX_SYNAPSE_WARPS);

	uint targetData =
	       ((uint(partition) & MASK(PARTITION_BITS)) << (DELAY_BITS + SYNAPSE_WARP_BITS))
	     | ((uint(delay)     & MASK(DELAY_BITS))     << (SYNAPSE_WARP_BITS))
	     |  (uint(warp)      & MASK(SYNAPSE_WARP_BITS));

	return make_uint4(targetData, (uint) warpOffset, warpTargetBits, 0);
}




__host__
void
setOutgoingPitch(size_t targetPitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_outgoingPitch,
				&targetPitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}



__host__ __device__
size_t
outgoingRow(pidx_t partition, nidx_t neuron, size_t pitch)
{
	//! \todo factor out addressing function and share with the 'counts' function
	return (partition * MAX_PARTITION_SIZE + neuron) * pitch;
}



__device__
uint
outgoingTargetPartition(outgoing_t out)
{
	return uint((out.x >> (DELAY_BITS + SYNAPSE_WARP_BITS)) & MASK(PARTITION_BITS));
}



__device__
uint
outgoingDelay(outgoing_t out)
{
	return uint((out.x >> SYNAPSE_WARP_BITS) & MASK(DELAY_BITS));
}



__device__
uint
outgoingWarp(outgoing_t out)
{
	return uint(out.x & MASK(SYNAPSE_WARP_BITS));
}



__device__
uint
outgoingWarpOffset(outgoing_t out)
{
	return out.y;
}



__device__
uint
outgoingTargetBits(outgoing_t out)
{
	return out.z;
}



__device__
outgoing_t
outgoing(uint presynaptic,
		uint jobIdx,
		outgoing_t* g_targets)
{
	size_t addr = outgoingRow(CURRENT_PARTITION, presynaptic, c_outgoingPitch);
	return g_targets[addr + jobIdx];
}



/*! \return
 *		the number of jobs for a particular firing neuron in the current
 *		partition */
__device__
uint
outgoingCount(uint presynaptic, uint* g_counts)
{
	return g_counts[CURRENT_PARTITION * MAX_PARTITION_SIZE + presynaptic];
}


#endif
