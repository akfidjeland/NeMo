#include "kernel.cu_h"
#include "connectivityMatrix.cu_h"

// format forward synapse: time - partition - neuron
// format reverse synapse: forward address - partition - neuron


#define NEURON_MASK (~(~0 << NEURON_BITS))
#define PARTITION_MASK (~(~0 << PARTITION_BITS))
#define FSYNAPSE_MASK (~(~0 << SYNAPSE_BITS))

#define PARTITION_SHIFT NEURON_BITS
#define FSYNAPSE_SHIFT (PARTITION_SHIFT + PARTITION_BITS)

#define ARRIVAL_SHIFT FSYNAPSE_SHIFT
#define ARRIVAL_MASK FSYNAPSE_MASK



__host__
uint
packSynapse(uint partition, uint neuron)
{
    assert(!(partition & ~PARTITION_MASK));
    assert(!(neuron & ~NEURON_MASK));
	// leave timestamp field at 0
    return (partition << PARTITION_SHIFT) | neuron;
}


__device__
uint
targetNeuron(uint synapse)
{
    return synapse & NEURON_MASK;
}


__device__
uint
targetPartition(uint synapse)
{
    return (synapse >> PARTITION_SHIFT) & PARTITION_MASK;
}


__device__
uint
lastSpikeArrival(uint2 synapse)
{
    return synapse.x >> ARRIVAL_SHIFT;
}


__device__
uint
setTimestamp(uint synapse, uint cycle)
{
	return ((cycle & ARRIVAL_MASK) << ARRIVAL_SHIFT)
		| (synapse & ~(ARRIVAL_MASK << ARRIVAL_SHIFT));
}


__device__
uint
arrivalTime(uint synapse)
{
	return (synapse >> ARRIVAL_SHIFT) & ARRIVAL_MASK;
}


__host__
uint
packReverseSynapse(
        uint sourcePartition,
        uint sourceNeuron,
        uint arrivalTime)
{
    assert(!(sourcePartition & ~PARTITION_MASK));
    assert(!(sourceNeuron & ~NEURON_MASK));
    /* The arrival time can wrap around */
    return ((arrivalTime & FSYNAPSE_MASK) << FSYNAPSE_SHIFT) 
            | (sourcePartition << PARTITION_SHIFT) 
            | sourceNeuron;
}



__device__
uint
sourceNeuron(uint rsynapse)
{
    return rsynapse & NEURON_MASK;
}


__device__
uint
sourcePartition(uint rsynapse)
{
    return (rsynapse >> PARTITION_SHIFT) & PARTITION_MASK;
}


__device__
uint
forwardIdx(uint rsynapse)
{
    return rsynapse >> FSYNAPSE_SHIFT;
}
