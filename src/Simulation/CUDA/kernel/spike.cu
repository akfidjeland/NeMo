#ifndef SPIKE_CU
#define SPIKE_CU

/* Packing and unpacking of L1 spikes spikes
 * 
 * For spike delivery we distinguish between L0 (short-range) and L1
 * (long-range) synapses.
 * 
 * For L0 delivery current is copied directly from the connectivity matrix to
 * the current accumulator. For L1, however, spike delivery is a two-step
 * process via a spike buffer written to and read from by separate partitions.
 *
 * Each such spike contains the weight plus address information. For regular
 * delivery only the target neuron index is needed. Both the source and target
 * partition indices are implicit in the organisation of the spike buffers.
 * 
 * If synapses are modified at run time, however, we may need to address the
 * spike in the forward matrix at the time when the spike is delivered. We thus
 * need the source neuron index as well as the source synapse index. 
 * 
 * Currently the address part of each spike is 32 bits. With plastic synapses
 * this places a limit in the connectivity. With a partition size of 1k, we can
 * support 4k synapses per neuron, whereas with a partition size of 2k we can
 * only support 1k synapses per neuron 
 *
 *
 * Format:
 *
 * (MSb) SSSSSSSDDDDDNNNNNNNNNNTTTTTTTTTT (LSb)
 *       |--7--||-5-||---10---||---10---|
 *
 * where
 *
 * S: source synapse index
 * D: delay
 * N: source neuron
 * T: target neuron
 */

#include "kernel.cu_h"

typedef uint2 spike_t;


#define SPIKE_SYNAPSE_BITS (32-(2*NEURON_BITS)-DELAY_BITS)
//! \todo remove this to utility header and share with connectivityMatrix
#define MASK(bits) (~(~0 << bits))
#define IN_RANGE(val, bits) ASSERT(!(val & ~MASK(bits)))


__device__
spike_t
packSpike_static(
		uint sourceNeuron,
		uint delay,
		uint sourceSynapse,
		uint targetNeuron,
		float weight)
{
	IN_RANGE(sourceNeuron, NEURON_BITS);
	uint address =
		targetNeuron;
	return make_uint2(address, __float_as_int(weight));
}

__device__
spike_t
packSpike_STDP(
		uint sourceNeuron,
		uint delay,
		uint sourceSynapse,
		uint targetNeuron,
		float weight)
{
	IN_RANGE(sourceNeuron, NEURON_BITS);
	IN_RANGE(targetNeuron, NEURON_BITS);
	IN_RANGE(delay, DELAY_BITS);
	IN_RANGE(sourceSynapse, SPIKE_SYNAPSE_BITS);
	uint address =
		sourceSynapse << (NEURON_BITS*2 + DELAY_BITS) |
		delay << (NEURON_BITS*2)                      |
		sourceNeuron << NEURON_BITS                   |
		targetNeuron;
	return make_uint2(address, __float_as_int(weight));
}


__device__
uint
spikeTargetNeuron(spike_t s)
{
	return s.x & MASK(NEURON_BITS);
}


__device__
uint
spikeSourceNeuron(spike_t s)
{
	return (s.x >> NEURON_BITS) & MASK(NEURON_BITS);
}


/* \return 0-based delay (shortest delay is 0, rather than 1, suitable for addressing) */
//! \todo store 1-based delays for clearer code
__device__
uint
spikeDelay(spike_t s)
{
	return (s.x >> (NEURON_BITS*2)) & MASK(DELAY_BITS);
}


__device__
uint
spikeSourceSynapse(spike_t s)
{
	return s.x >> (NEURON_BITS*2+DELAY_BITS);
	
}


__device__
float
spikeWeight(spike_t s)
{
	return __int_as_float(s.y);	
}

#endif
