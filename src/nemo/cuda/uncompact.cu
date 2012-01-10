#include <nemo/config.h>

/*! Convert from fully compacted firing to per-partition compacted firing
 *
 * This kernel makes no assumptions about ordering of the input, and could
 * therefore probably be improved. 
 */
__global__
void
uncompact(
		param_t* g_params,
		unsigned ncompact[],
		unsigned sz_ncompact,
		nidx_dt pcompact[])
{
	__shared__ unsigned s_pfill[MAX_PARTITION_COUNT]; 

	if(threadIdx.x < MAX_PARTITION_COUNT) {
		s_pfill[threadIdx.x] = 0U;
	}
	__syncthreads();

	for(unsigned b = 0 ; b < sz_ncompact; b += blockDim.x) {
		// read neuron
		unsigned fired = ncompact[b + threadIdx.x];
		nidx_dt n = fired & MASK(NEURON_BITS);
		pidx_t p = fired >> NEURON_BITS;
		ASSERT(p < MAX_PARTITION_COUNT);
		/*! \todo The most likely scenario is that the compact list of neurons
		 * is ordered. Groups of neighbouring data share the same partition, so
		 * using atomics here is very inefficient. Fix this! */
		unsigned i = atomicAdd(s_pfill + p, 1);
		pcompact[p * g_params->pitch32 + i] = n;
	}
}


__host__
cudaError_t
uncompact(cudaStream_t stream,
		unsigned partitionCount,
		param_t* d_params,
		unsigned ncompact[],
		unsigned sz_ncompact,
		nidx_dt pcompact[])
{
	dim3 dimBlock(1024);
	dim3 dimGrid(partitionCount);
	uncompact<<<dimGrid, dimBlock, 0, stream>>>(d_params, ncompact, sz_ncompact, pcompact);
	return cudaGetLastError();
}
