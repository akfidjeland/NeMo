#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

#include "outgoing.cu_h"

__constant__ size_t c_outgoingPitch; // word pitch


__host__
outgoing_t
make_outgoing(pidx_t partition, delay_t delay, uint warps)
{
	assert(partition < MAX_PARTITION_COUNT);
	assert(delay < MAX_DELAY);
	assert(warps < MAX_SYNAPSE_WARPS);
	assert(MAX_PARTITION_COUNT < 256);
	assert(MAX_DELAY < 256);
	assert(MAX_SYNAPSE_WARPS <= 256);
	return make_uchar4((uchar) partition, (uchar) delay, (uchar) warps, 0);
}



__host__
bool
operator<(const outgoing_t& a, const outgoing_t& b)
{
	return a.x < b.x || (a.x == b.x && a.y < b.y);
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
	return (uint) out.x;
}



__device__
uint
outgoingDelay(outgoing_t out)
{
	return (uint) out.y;
}



__device__
uint
outgoingWarps(outgoing_t out)
{
	return (uint) out.z;
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
