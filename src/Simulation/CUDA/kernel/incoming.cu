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



__device__
incoming_t
getIncoming(uint cycle, uint offset, incoming_t* g_incoming)
{
	return g_incoming[incomingBufferStart(CURRENT_PARTITION, cycle, 0) + offset];
}




/*! \return incoming spike group from a particular source */
__device__
incoming_t
make_incoming(uint sourcePartition, uint sourceNeuron, uint delay, uint warps, void* address)
{
	ASSERT(sourcePartition < (1<<PARTITION_BITS));
	ASSERT(sourceNeuron < (1<<NEURON_BITS));
	ASSERT(delay < (1<<DELAY_BITS));
	ASSERT(warps < (1<<SYNAPSE_WARP_BITS));

	uint sourceData =
	       ((uint(sourcePartition) & MASK(PARTITION_BITS)) << (SYNAPSE_WARP_BITS + DELAY_BITS + NEURON_BITS))
	     | ((uint(sourceNeuron)    & MASK(NEURON_BITS))    << (SYNAPSE_WARP_BITS + DELAY_BITS))
	     | ((uint(delay)           & MASK(DELAY_BITS))     << (SYNAPSE_WARP_BITS))
	     | ((uint(warps)           & MASK(SYNAPSE_WARP_BITS )));

	uint64_t ptr64 = (uint64_t) address;

#ifdef __DEVICE_EMULATION__
	uint32_t low = (uint32_t) (ptr64 & 0xffffffff);
	uint32_t high = (uint32_t) ((ptr64 >> 32) & 0xffffffff);
	return make_uint4(sourceData, (uint) low, (uint) high, 0);
#else
	const uint64_t MAX_ADDRESS = 4294967296LL; // on device
	ASSERT(ptr64 < MAX_ADDRESS);
	return make_uint2(sourceData, (uint) ptr64);
#endif
}


__device__
uint
incomingDelay(incoming_t in)
{
	return (in.x >> SYNAPSE_WARP_BITS) & MASK(DELAY_BITS);
}


__device__
uint
incomingPartition(incoming_t in)
{
	return (in.x >> (SYNAPSE_WARP_BITS + DELAY_BITS + NEURON_BITS)) & MASK(PARTITION_BITS);
}



__device__
uint
incomingNeuron(incoming_t in)
{
	return (in.x >> (SYNAPSE_WARP_BITS + DELAY_BITS)) & MASK(NEURON_BITS);
}



__device__
uint
incomingWarps(incoming_t in)
{
	return in.x & MASK(SYNAPSE_WARP_BITS);
}



/*! \return address into matrix with number of incoming synapse groups */
__device__
size_t
incomingCountAddr(uint targetPartition, uint cycle, uint delay1)
{
	return targetPartition * MAX_DELAY + incomingSlot(cycle, delay1);
}



#endif
