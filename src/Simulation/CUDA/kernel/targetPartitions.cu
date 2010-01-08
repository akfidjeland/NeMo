#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

#include "targetPartitions.cu_h"

__constant__ size_t c_targetpPitch; // word pitch


__host__
targetp_t
make_targetp(pidx_t partition, delay_t delay)
{
	assert(partition < MAX_PARTITION_COUNT);
	assert(delay < MAX_DELAY);
	assert(MAX_PARTITION_COUNT < 256);
	assert(MAX_DELAY < 256);
	return make_uchar2((uchar) partition, (uchar) delay);
}



__host__
bool
operator<(const targetp_t& a, const targetp_t& b)
{
	return a.x < b.x || (a.x == b.x && a.y < b.y);
}



__host__
void
setTargetPitch(size_t targetPitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_targetpPitch,
				&targetPitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}



__host__ __device__
size_t
targetIdx(
		pidx_t partition,
		nidx_t neuron,
		size_t pitch)
{
	//! \todo factor out addressing function and share with the 'counts' function
	return (partition * MAX_PARTITION_SIZE + neuron) * pitch;
}



__device__
uint
job_targetPartition(targetp_t job)
{
	return (uint) job.x;
}



__device__
uint
job_delay(targetp_t job)
{
	return (uint) job.y;
}



__device__
targetp_t
targetPartitions(uint presynaptic,
		uint jobIdx,
		targetp_t* g_targets)
{
	size_t addr = targetIdx(CURRENT_PARTITION, presynaptic, c_targetpPitch);
	return g_targets[addr + jobIdx];
}



/*! \return the number of jobs for a particular firing neuron in the current partition */
__device__
uint
jobCount(uint presynaptic, uint* g_jobCounts)
{
	return g_jobCounts[CURRENT_PARTITION * MAX_PARTITION_SIZE + presynaptic];
}


#endif
