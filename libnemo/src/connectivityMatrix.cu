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

#include "kernel.cu_h"
#include "connectivityMatrix.cu_h"
#include "util.h"

#define NEURON_MASK MASK(NEURON_BITS)
#define PARTITION_MASK MASK(PARTITION_BITS)
#define FSYNAPSE_MASK MASK(SYNAPSE_BITS)
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
void
setFcmPlaneSize(size_t sz)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_fcmPlaneSize,
				&sz, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}


__host__
uint
f_packSynapse(uint neuron)
{
	return neuron;
}


__host__ __device__
uint
targetNeuron(uint synapse)
{
#ifdef __DEVICE_EMULATION__
    return synapse & NEURON_MASK;
#else
	return synapse;
#endif
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



__device__ __host__
uint
sourceNeuron(uint rsynapse)
{
    return (rsynapse >> R_NEURON_SHIFT) & NEURON_MASK;
}


__device__ __host__
uint
sourcePartition(uint rsynapse)
{
    return (rsynapse >> R_PARTITION_SHIFT) & PARTITION_MASK;
}


__device__ __host__
uint
forwardIdx(uint rsynapse)
{
    return rsynapse >> R_FSYNAPSE_SHIFT;
}



__device__ __host__
uint
r_delay1(uint rsynapse)
{
    return rsynapse & DELAY_MASK; 
}


__device__
uint
r_delay0(uint rsynapse)
{
	return r_delay1(rsynapse) - 1;
}



/* To improve packing of data in the connectivity matrix, we use different
 * pitches for each partition */
//! \todo store offset instead of pointers
__constant__ DEVICE_UINT_PTR_T cr_pitch[MAX_THREAD_BLOCKS];

/* We also need to store the start of each partitions reverse connectivity
 * data, to support fast lookup. This data should nearly always be in the
 * constant cache */
__constant__ DEVICE_UINT_PTR_T cr_address[MAX_THREAD_BLOCKS];

/* Ditto for the STDP accumulators */
__constant__ DEVICE_UINT_PTR_T cr_stdp[MAX_THREAD_BLOCKS];

/* Ditto for the forward synapse offset */
__constant__ DEVICE_UINT_PTR_T cr_faddress[MAX_THREAD_BLOCKS];



#define SET_CR_ADDRESS_VECTOR(symbol, vec, len) CUDA_SAFE_CALL(\
		cudaMemcpyToSymbol(symbol, vec, len * sizeof(DEVICE_UINT_PTR_T), 0, cudaMemcpyHostToDevice)\
	)



__host__
void
configureReverseAddressing(
        DEVICE_UINT_PTR_T* r_pitch,
        DEVICE_UINT_PTR_T* r_address,
        DEVICE_UINT_PTR_T* r_stdp,
        DEVICE_UINT_PTR_T* r_faddress,
		size_t len)
{
	//! \todo extend vectors and fill with NULLs
	SET_CR_ADDRESS_VECTOR(cr_pitch, r_pitch, len);
	SET_CR_ADDRESS_VECTOR(cr_address, r_address, len);
	SET_CR_ADDRESS_VECTOR(cr_stdp, r_stdp, len);
	SET_CR_ADDRESS_VECTOR(cr_faddress, r_faddress, len);
}

#endif
