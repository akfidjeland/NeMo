#ifndef TARGET_PARTITIONS_CU
#define TARGET_PARTITIONS_CU

__constant__ size_t c_targetpPitch;


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
