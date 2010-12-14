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



/* distance (in words) between a synapses's address data and its weight data. */
__constant__ size_t c_fcmPlaneSize;

__host__
synapse_t
f_nullSynapse()
{
	return 0;
}


__host__
cudaError
setFcmPlaneSize(size_t sz)
{
	return cudaMemcpyToSymbol(c_fcmPlaneSize,
				&sz, sizeof(size_t), 0, cudaMemcpyHostToDevice);
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
unsigned
r_packSynapse(unsigned sourcePartition, unsigned sourceNeuron, unsigned delay)
{
	assert(!(sourcePartition & ~PARTITION_MASK));
	assert(!(sourceNeuron & ~NEURON_MASK));
	assert(!(delay & ~DELAY_MASK));
	return (sourcePartition << R_PARTITION_SHIFT)
	     | (sourceNeuron    << R_NEURON_SHIFT)
	     |  delay;
}



__device__ __host__
unsigned
sourceNeuron(unsigned rsynapse)
{
    return (rsynapse >> R_NEURON_SHIFT) & NEURON_MASK;
}


__device__ __host__
unsigned
sourcePartition(unsigned rsynapse)
{
    return (rsynapse >> R_PARTITION_SHIFT) & PARTITION_MASK;
}



__device__ __host__
unsigned
r_delay1(unsigned rsynapse)
{
    return rsynapse & DELAY_MASK; 
}


__device__
unsigned
r_delay0(unsigned rsynapse)
{
	return r_delay1(rsynapse) - 1;
}



/* To improve packing of data in the connectivity matrix, we use different
 * pitches for each partition */
__constant__ size_t cr_pitch[MAX_PARTITION_COUNT];

/* We also need to store the start of each partitions reverse connectivity
 * data, to support fast lookup. This data should nearly always be in the
 * constant cache */
__constant__ uint32_t* cr_address[MAX_PARTITION_COUNT];

/* Ditto for the STDP accumulators */
__constant__ weight_dt* cr_stdp[MAX_PARTITION_COUNT];

/* Ditto for the forward synapse offset */
__constant__ uint32_t* cr_faddress[MAX_PARTITION_COUNT];



#define SET_CR_ADDRESS_VECTOR(symbol, vec, len, type)                         \
    err = cudaMemcpyToSymbol(symbol, vec,                                     \
            len * sizeof(type), 0, cudaMemcpyHostToDevice);                   \
    if(cudaSuccess != err) {                                                  \
        return err;                                                           \
    }



__host__
cudaError
configureReverseAddressing(
        size_t* r_pitch,
        uint32_t* const* r_address,
        weight_dt* const* r_stdp,
        uint32_t* const* r_faddress,
		size_t len)
{
	cudaError err;
	//! \todo extend vectors and fill with NULLs
	SET_CR_ADDRESS_VECTOR(cr_pitch, r_pitch, len, size_t);
	SET_CR_ADDRESS_VECTOR(cr_address, r_address, len, uint32_t*);
	SET_CR_ADDRESS_VECTOR(cr_stdp, r_stdp, len, weight_dt*);
	SET_CR_ADDRESS_VECTOR(cr_faddress, r_faddress, len, uint32_t*);
	return cudaSuccess;
}

#endif
