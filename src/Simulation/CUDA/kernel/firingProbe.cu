#include "kernel.cu_h"


//! \todo could use OUTPUT_BUFFER_SZ instead of pitch here
/* \param g_firingOutput
 *      Firing output buffer in global memory, offset by 1) cycle and 2)
 *      partition
 * \param pitch
 *      Number of 32-bit words per partition in the firing buffer
 */
__device__
void
writeFiringOutput(uint nfired,
		dnidx_t* s_fired,
		uint32_t* s_dfired, // dense firing
		size_t pitch, uint32_t* g_firingOutput)
{
	// clear
	if(threadIdx.x < (MAX_PARTITION_SIZE/32)) {
		s_dfired[threadIdx.x] = 0;
	}
	__syncthreads();

	// fill
	for(uint nbase=0; nbase < nfired; nbase += THREADS_PER_BLOCK) {
		uint i = nbase + threadIdx.x;
		uint neuron = s_fired[i];
		ASSERT(neuron / 32 < OUTPUT_BUFFER_SZ);
		uint32_t word = neuron / 32;
		uint32_t mask = 0x1 << (neuron % 32);
		if(i < nfired) {
			atomicOr(s_dfired + word, mask);
		}
	}
	__syncthreads();

	// write to global memory
	if(threadIdx.x < pitch) {
		g_firingOutput[threadIdx.x] =  s_dfired[threadIdx.x];
	}
}




/*! \return Did the given neuron fire this cycle? */
__device__
uint32_t
didFire(uint neuron, uint32_t* s_dfired)
{
	//! \todo check that we're in bounds
	uint32_t word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	return s_dfired[word] & mask;
}

