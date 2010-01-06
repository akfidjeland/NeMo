#ifndef L1_SPIKE_BUFFER_CU
#define L1_SPIKE_BUFFER_CU

#include "kernel.cu_h"
#include "l1SpikeBuffer.cu_h"

__constant__ size_t c_l1BufferPitch;


__host__
void
setBufferPitch(size_t pitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_l1BufferPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}


#endif
