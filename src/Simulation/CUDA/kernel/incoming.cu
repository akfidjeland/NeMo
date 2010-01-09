#ifndef L1_SPIKE_BUFFER_CU
#define L1_SPIKE_BUFFER_CU

#include "kernel.cu_h"
#include "incoming.cu_h"


__constant__ size_t c_incomingPitch; // word pitch


__host__
void
setIncomingPitch(size_t pitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_incomingPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}



/*! \return the buffer number to use for the given delay, given current cycle */
__device__
uint
incomingSlot(uint cycle, uint delay1)
{
	return (cycle + delay1) % MAX_DELAY;
}



/* Return offset into full buffer data structure to beginning of buffer for a
 * particular targetPartition and a particular delay. */
__device__
uint
incomingBufferStart(uint targetPartition, uint cycle, uint delay1)
{
	return (targetPartition * MAX_DELAY + incomingSlot(cycle, delay1)) * c_incomingPitch;
}



/*! \return incoming spike group from a particular source */
__device__
incoming_t
make_incoming(uint sourcePartition, uint sourceNeuron, uint delay)
{
	ASSERT(sourcePartition < (1<<8));
	ASSERT(sourceNeuron < (1<<16));
	ASSERT(delay < (1<<8));
	return make_uchar4(
			(uchar) sourcePartition,
			(uchar) sourceNeuron >> 8,   // MSB
			(uchar) sourceNeuron & 0xff, // LSB
			(uchar) delay);
}


#endif
