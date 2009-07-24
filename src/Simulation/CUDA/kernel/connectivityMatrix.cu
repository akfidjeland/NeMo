#include <assert.h>

#include "kernel.cu_h"
#include "connectivityMatrix.cu_h"

//! \todo factor out MASK logic
#define NEURON_MASK (~(~0 << NEURON_BITS))
#define PARTITION_MASK (~(~0 << PARTITION_BITS))
#define FSYNAPSE_MASK (~(~0 << SYNAPSE_BITS))
#define DELAY_MASK (~(~0 << DELAY_BITS))

#define PARTITION_SHIFT NEURON_BITS

/* Reverse synapses */
#define R_FSYNAPSE_SHIFT (R_PARTITION_SHIFT + PARTITION_BITS)
#define R_PARTITION_SHIFT (R_NEURON_SHIFT + NEURON_BITS)
#define R_NEURON_SHIFT DELAY_BITS



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
        uint sourceSynapse,
        uint delay)
{
    assert(!(sourcePartition & ~PARTITION_MASK));
    assert(!(sourceNeuron & ~NEURON_MASK));
    assert(!(delay & ~DELAY_MASK));
    return (  (sourceSynapse & FSYNAPSE_MASK) << R_FSYNAPSE_SHIFT)
            | (sourcePartition                << R_PARTITION_SHIFT)
            | (sourceNeuron                   << R_NEURON_SHIFT)
            | delay;
}



__device__
uint
sourceNeuron(uint rsynapse)
{
    return (rsynapse >> R_NEURON_SHIFT) & NEURON_MASK;
}


__device__
uint
sourcePartition(uint rsynapse)
{
    return (rsynapse >> R_PARTITION_SHIFT) & PARTITION_MASK;
}


__device__
uint
forwardIdx(uint rsynapse)
{
    return rsynapse >> R_FSYNAPSE_SHIFT;
}


__device__
uint
r_delay(uint rsynapse)
{
    return rsynapse & DELAY_MASK; 
}
