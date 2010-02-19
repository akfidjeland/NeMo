#ifndef LIBNEMO_H
#define LIBNEMO_H

/* ! \todo move partition sizing from haskell code to c code. Then we'll no
 * longer need this include. */
#include "kernel.cu_h"

#include <string.h> // for size_t


typedef int status_t;

#define KERNEL_OK 0
#define KERNEL_INVOCATION_ERROR 1
#define KERNEL_ASSERTION_FAILURE 2
//! \todo add additional errors for memory allocation etc.


//-----------------------------------------------------------------------------
// RUNTIME DATA
//-----------------------------------------------------------------------------


/* Opaque pointer to network object */
typedef void* RTDATA;


//! \todo get rid of max partition size argument. This is only really useful for debugging.
//! \todo get rid of setReverse. Just deal with this in the host code
//! \todo dynamically resize the firing buffer?
/*!
 * \param maxPartitionSize
 * \param setReverse
 * 		set reverse connectivity matrix, required e.g. for STDP
 * \param maxReadPeriod
 * 		maximum period (in cycles) between reads to the device firing buffer
 */
RTDATA
nemo_new_network(
		size_t maxPartitionSize,
		unsigned int setReverse,
		unsigned int maxReadPeriod);

void nemo_delete_network(RTDATA);


//-----------------------------------------------------------------------------
// LOADING NEURON DATA
//-----------------------------------------------------------------------------


//! \todo should return error status here
void
nemo_add_neuron(RTDATA,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);



//-----------------------------------------------------------------------------
// LOADING SYNAPSE DATA
//-----------------------------------------------------------------------------


//! \todo should return error status here
void
nemo_add_synapses(RTDATA,
		unsigned int source,
		unsigned int targets[],
		unsigned int delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length);



/*! Read connectivity matrix back from device for a single neuron and delay. */
size_t
nemo_get_synapses(RTDATA rtdata,
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetPartition[],
		unsigned int* targetNeuron[],
		float* weights[],
		unsigned char* plastic[]);

//-----------------------------------------------------------------------------
// FIRING PROBE
//-----------------------------------------------------------------------------


/*! Return all buffered firing data and empty buffers.
 *
 * The three arrays together form a vector of 3-tuples specifying cycle,
 * partition index, and neuron index for all the fired neurons. 
 *
 * The last two output variables contain the number of neurons and the number of
 * cycles for which we have firing.
 */
void
nemo_read_firing(RTDATA rtdata,
		unsigned int** cycles,
		unsigned int** neuronIdx,
		unsigned int* nfired,
		unsigned int* ncycles);


/* Step can be asynchronous. sync forces completion of all steps */
void nemo_sync_simulation(RTDATA rtdata);


/* If the user is not reading back firing, the firing output buffers should be
 * flushed to avoid buffer overflow. The overflow is not harmful in that no
 * memory accesses take place outside the buffer, but an overflow may result in
 * later calls to readFiring returning non-sensical results. */
void nemo_flush_firing_buffer(RTDATA rtdata);


//-----------------------------------------------------------------------------
// TIMING
//-----------------------------------------------------------------------------

void nemo_print_cycle_counters(RTDATA rtdata);


/*! \return number of milliseconds elapsed between beginning of first kernel
 * invocation and the end of the last */
long int nemo_elapsed_ms(RTDATA rtdata);


void
nemo_reset_timer(RTDATA rtdata);


//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


/*! Enable spike-timing dependent plasticity in the simulation.
 *
 * \param preFireLen
 * 		Length, in cycles, of the part of the STDP window that precedes the
 * 		postsynaptic firing.
 * \param postFireLen
 * 		Length, in cycles, of the part of the STDP window that comes after the
 * 		postsynaptic firing.
 * \param potentiationMask
 * 		Bit mask indicating what cycles during the STDP for which potentiation
 * 		takes place. Bit 0 is the end of the STDP window.
 * \param depressionMask
 * 		Bit mask indicating what cycles during the STDP for which depression
 * 		takes place. Bit 0 is the end of the STDP window.
 * \param preFireFn
 * 		STDP function sampled at integer cycle intervals in the prefire part of
 * 		the STDP window
 * \param preFireFn
 * 		STDP function sampled at integer cycle intervals in the postfire part of
 * 		the STDP window
 * \param maxWeight
 * 		Weight beyond which excitatory synapses are not allowed to move
 * \param minWeight
 * 		Weight beyond which excitatory synapses are not allowed to move
 */
void
nemo_enable_stdp(RTDATA,
		unsigned int preFireLen,
		unsigned int postFireLen,
		float* preFireFn,
		float* postFireFn,
		float maxWeight,
		float minWeight);


//-----------------------------------------------------------------------------
// SIMULATION STEPPING
//-----------------------------------------------------------------------------

status_t
nemo_step(RTDATA rtdata,
		int substeps,               // number of substeps per normal 1ms step
		// External firing (sparse)
		size_t extFiringCount,
		const int* extFiringNIdx);  // neuron indices


void nemo_apply_stdp(RTDATA rtdata, float stdpReward);


/* Force all allocated memory onto the device. Calling this is not required
 * during normal operation, as step invokes it on first call, but can be used
 * for testing */
void
nemo_start_simulation(RTDATA);


//-----------------------------------------------------------------------------
// DEVICE PROPERTIES
//-----------------------------------------------------------------------------

//! \return number of cuda-enabled devices of compute capability 1.0 or greater
int nemo_device_count(void);


#endif
