#ifndef CONNECTIVITY_MATRIX_CU
#define CONNECTIVITY_MATRIX_CU

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
void
setFcmPlaneSize(size_t sz)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_fcmPlaneSize,
				&sz, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}


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
__constant__ DEVICE_UINT_PTR_T cr0_pitch[MAX_THREAD_BLOCKS];
__constant__ DEVICE_UINT_PTR_T cr1_pitch[MAX_THREAD_BLOCKS];

/* We also need to store the start of each partitions reverse connectivity
 * data, to support fast lookup. This data should nearly always be in the
 * constant cache */
__constant__ DEVICE_UINT_PTR_T cr0_address[MAX_THREAD_BLOCKS];
__constant__ DEVICE_UINT_PTR_T cr1_address[MAX_THREAD_BLOCKS];

/* Ditto for the STDP accumulators */
__constant__ DEVICE_UINT_PTR_T cr0_stdp[MAX_THREAD_BLOCKS];
__constant__ DEVICE_UINT_PTR_T cr1_stdp[MAX_THREAD_BLOCKS];

/* Ditto for the forward synapse offset */
__constant__ DEVICE_UINT_PTR_T cr0_faddress[MAX_THREAD_BLOCKS];
__constant__ DEVICE_UINT_PTR_T cr1_faddress[MAX_THREAD_BLOCKS];


#define SET_CR_ADDRESS_VECTOR(symbol, vec) CUDA_SAFE_CALL(\
		cudaMemcpyToSymbol(symbol, &vec[0], vec.size() * sizeof(DEVICE_UINT_PTR_T), 0, cudaMemcpyHostToDevice)\
	)




__host__
void
configureReverseAddressing(
        const std::vector<DEVICE_UINT_PTR_T>& r0_pitch,
        const std::vector<DEVICE_UINT_PTR_T>& r0_address,
        const std::vector<DEVICE_UINT_PTR_T>& r0_stdp,
        const std::vector<DEVICE_UINT_PTR_T>& r0_faddress,
        const std::vector<DEVICE_UINT_PTR_T>& r1_pitch,
        const std::vector<DEVICE_UINT_PTR_T>& r1_address,
        const std::vector<DEVICE_UINT_PTR_T>& r1_stdp,
        const std::vector<DEVICE_UINT_PTR_T>& r1_faddress)
{
	//! \todo extend vectors and fill with NULLs
	SET_CR_ADDRESS_VECTOR(cr0_pitch, r0_pitch);
	SET_CR_ADDRESS_VECTOR(cr0_address, r0_address);
	SET_CR_ADDRESS_VECTOR(cr0_stdp, r0_stdp);
	SET_CR_ADDRESS_VECTOR(cr0_faddress, r0_faddress);
	SET_CR_ADDRESS_VECTOR(cr1_pitch, r1_pitch);
	SET_CR_ADDRESS_VECTOR(cr1_address, r1_address);
	SET_CR_ADDRESS_VECTOR(cr1_stdp, r1_stdp);
	SET_CR_ADDRESS_VECTOR(cr1_faddress, r1_faddress);
}

#endif
