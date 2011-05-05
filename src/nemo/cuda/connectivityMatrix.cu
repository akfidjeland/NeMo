#ifndef CONNECTIVITY_MATRIX_CU
#define CONNECTIVITY_MATRIX_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>

#include <nemo/util.h>

#include "kernel.cu_h"
#include "connectivityMatrix.cu_h"

#define NEURON_MASK MASK(NEURON_BITS)
#define PARTITION_MASK MASK(PARTITION_BITS)
#define DELAY_MASK MASK(DELAY_BITS)

#define PARTITION_SHIFT NEURON_BITS

/* Reverse synapses */
#define R_FSYNAPSE_SHIFT (R_PARTITION_SHIFT + PARTITION_BITS)
#define R_PARTITION_SHIFT (R_NEURON_SHIFT + NEURON_BITS)
#define R_NEURON_SHIFT DELAY_BITS


__host__
synapse_t
f_nullSynapse()
{
	return 0;
}



__host__ __device__
unsigned
targetNeuron(unsigned synapse)
{
#ifdef __DEVICE_EMULATION__
    return synapse & NEURON_MASK;
#else
	return synapse;
#endif
}


__host__
rsynapse_t
r_packSynapse(unsigned sourcePartition, unsigned sourceNeuron, unsigned delay)
{
	assert(!(sourcePartition & ~PARTITION_MASK));
	assert(!(sourceNeuron & ~NEURON_MASK));
	assert(!(delay & ~DELAY_MASK));
	rsynapse_t s = 0;
	s |= sourcePartition << R_PARTITION_SHIFT;
	s |= sourceNeuron    << R_NEURON_SHIFT;
	s |= delay;
	return s;
}



__device__ __host__
unsigned
sourceNeuron(rsynapse_t rsynapse)
{
    return (rsynapse >> R_NEURON_SHIFT) & NEURON_MASK;
}


__device__ __host__
unsigned
sourcePartition(rsynapse_t rsynapse)
{
    return (rsynapse >> R_PARTITION_SHIFT) & PARTITION_MASK;
}



__device__ __host__
unsigned
r_delay1(rsynapse_t rsynapse)
{
    return rsynapse & DELAY_MASK; 
}


__device__
unsigned
r_delay0(rsynapse_t rsynapse)
{
	return r_delay1(rsynapse) - 1;
}

#endif
