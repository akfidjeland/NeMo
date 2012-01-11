#include <nemo/config.h>

/*! Compact a boolean vector of firing to the per-partition compact
 * representation used internally.
 *
 * \param g_fired
 *		global memory with one entry per neuron with non-zero words indicating
 *		fired neurons.
 * \param nNeurons
 *		number of entries in \a g_fired
 * \param g_firedCompact
 *		global memory with one entry per neuron, but compacted on a
 *		per-partition basis. Non-zero entries are neuron indices of fired
 *		neurons. The partition index is implicit in the data structure.
 */
__global__
void
compact(unsigned g_fired[],
		unsigned nNeurons,
		nidx_dt g_firedCompact[])
{
	__shared__ nidx_dt s_firedCompact[MAX_PARTITION_SIZE];
	__shared__ unsigned s_nFired;

	for(unsigned bNeuron = 0; bNeuron < nNeurons; bNeuron += blockIdx.x) {
		unsigned neuron = bNeuron + threadIdx.x;
		s_firedCompact[neuron] = 0;
	}
	if(threadIdx.x == 0) {
		s_nFired = 0;
	}
	__syncthreads();

	for(unsigned bNeuron = 0; bNeuron < MAX_PARTITION_SIZE; bNeuron += blockIdx.x) {
		unsigned neuron = bNeuron + threadIdx.x;
		if(g_fired[neuron]) {
			unsigned i = atomicAdd(&s_nFired, 1);
			s_firedCompact[i] = neuron;
		}
	}
	__syncthreads();

	for(unsigned bNeuron = 0; bNeuron < MAX_PARTITION_SIZE; bNeuron += blockIdx.x) {
		unsigned neuron = bNeuron + threadIdx.x;
		g_firedCompact[neuron] = s_firedCompact[neuron];
	}
}


__host__
cudaError_t
compact(cudaStream_t stream,
		unsigned partitionCount,
		unsigned d_fired[],
		unsigned nNeurons,
		nidx_dt d_firedCompact[])
{
	dim3 dimBlock(1024);
	dim3 dimGrid(partitionCount);
	compact<<<dimGrid, dimBlock, 0, stream>>>(d_fired, nNeurons, d_firedCompact);
	return cudaGetLastError();
}
