#include "kernel.cu_h"

#define OUTPUT_BUFFER_SZ (1<<(NEURON_BITS-5))

__shared__ uint32_t s_firingOutput[OUTPUT_BUFFER_SZ];


__device__
void
clearFiringOutput()
{
	if(threadIdx.x < OUTPUT_BUFFER_SZ) {
		s_firingOutput[threadIdx.x] = 0;
	}
	__syncthreads();
}



/* \param g_firingOutput
 *      Firing output buffer in global memory, offset by 1) cycle and 2)
 *      partition
 * \param pitch
 *      Number of 32-bit words per partition in the firing buffer
 */
__device__
void
writeFiringOutput(uint32_t* g_firingOutput, size_t pitch)
{
	if(threadIdx.x < pitch) {
		g_firingOutput[threadIdx.x] =  s_firingOutput[threadIdx.x];
	}
}



__device__
void
setFiringOutput(uint neuron)
{
	ASSERT(neuron / 32 < OUTPUT_BUFFER_SZ);
	atomicOr(s_firingOutput + neuron / 32, 0x1 << (neuron % 32));
}



/*! \return Did the given neuron fire this cycle? */
__device__
uint32_t
didFire(uint neuron)
{
	//! \todo check that we're in bounds
	uint32_t word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	return s_firingOutput[word] & mask;
}



#undef OUTPUT_BUFFER_SZ
