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
make_incoming(uint sourcePartition, uint sourceNeuron, uint delay)
{
	ASSERT(sourcePartition < (1<<PARTITION_BITS));
	ASSERT(sourceNeuron < (1<<NEURON_BITS));
	ASSERT(delay < (1<<DELAY_BITS));

	return (incoming_t) { sourcePartition, sourceNeuron, delay };
}


__device__
uint
incomingDelay(incoming_t in)
{
	return in.delay;
}


__device__
uint
incomingPartition(incoming_t in)
{
	return in.source_partition;
}



__device__
uint
incomingNeuron(incoming_t in)
{
	return in.source_neuron;
}



/*! \return address into matrix with number of incoming synapse groups */
__device__
size_t
incomingCountAddr(uint targetPartition, uint cycle, uint delay1)
{
	return targetPartition * MAX_DELAY + incomingSlot(cycle, delay1);
}


#endif
