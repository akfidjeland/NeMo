#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

#include "outgoing.cu_h"

__constant__ size_t c_outgoingPitch; // word pitch



__host__
outgoing_t
make_outgoing(pidx_t partition, delay_t delay, uint warp, void* address)
{
	//! \todo could share pointer packing with dispatchTable code
	assert(partition < MAX_PARTITION_COUNT);
	assert(delay < MAX_DELAY);
	assert(warp < MAX_SYNAPSE_WARPS);

	uint targetData =
	       ((uint(partition) & MASK(PARTITION_BITS)) << (DELAY_BITS + SYNAPSE_WARP_BITS))
	     | ((uint(delay)     & MASK(DELAY_BITS))     << (SYNAPSE_WARP_BITS))
	     |  (uint(warp)      & MASK(SYNAPSE_WARP_BITS));

	assert(sizeof(address) <= sizeof(uint64_t));

	uint64_t ptr64 = (uint64_t) address;

#ifdef __DEVICE_EMULATION__
	uint32_t low = (uint32_t) (ptr64 & 0xffffffff);
	uint32_t high = (uint32_t) ((ptr64 >> 32) & 0xffffffff);
	return make_uint4(targetData, (uint) low, (uint) high, 0);
#else
	const uint64_t MAX_ADDRESS = 4294967296LL; // on device
	assert(ptr64 < MAX_ADDRESS);
	return make_uint2(targetData, (uint) ptr64);
#endif
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
uint32_t*
outgoingWarpPointer(outgoing_t out)
{
#ifdef __DEVICE_EMULATION__
	uint64_t ptr = out.z;
	ptr <<= 32;
	ptr |= out.y;
	return (uint32_t*) ptr;
#else
	return (uint32_t*) out.y;
#endif
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
