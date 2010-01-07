#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

#include "targetPartitions.cu_h"

__constant__ size_t c_targetpPitch;


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
		delay_t delay,
		size_t partitionSize,
		size_t pitch)
{
	return ((partition * partitionSize + neuron) * MAX_DELAY + delay) * pitch;
}


#endif
