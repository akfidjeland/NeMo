#include "kernel.cu_h"
#include "connectivityMatrix.cu_h"

#define NEURON_MASK (~(~0 << NEURON_BITS))
#define PARTITION_MASK (~(~0 << PARTITION_BITS))
#define FSYNAPSE_MASK (~(~0 << SYNAPSE_BITS))

#define PARTITION_SHIFT NEURON_BITS
#define FSYNAPSE_SHIFT (PARTITION_SHIFT + PARTITION_BITS)

#define ARRIVAL_SHIFT FSYNAPSE_SHIFT
#define ARRIVAL_MASK FSYNAPSE_MASK



__host__
uint
f_packSynapse(uint partition, uint neuron)
{
    assert(!(partition & ~PARTITION_MASK));
    assert(!(neuron & ~NEURON_MASK));
    return (partition << PARTITION_SHIFT) | neuron;
}


__host__ __device__
uint
targetNeuron(uint synapse)
{
    return synapse & NEURON_MASK;
}


__host__ __device__
uint
targetPartition(uint synapse)
{
    return (synapse >> PARTITION_SHIFT) & PARTITION_MASK;
}


__host__
uint
r_packSynapse(
        uint sourcePartition,
        uint sourceNeuron,
        uint sourceSynapse)
{
    assert(!(sourcePartition & ~PARTITION_MASK));
    assert(!(sourceNeuron & ~NEURON_MASK));
    return (  (sourceSynapse & FSYNAPSE_MASK) << FSYNAPSE_SHIFT)
            | (sourcePartition                << PARTITION_SHIFT)
            |  sourceNeuron;
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
