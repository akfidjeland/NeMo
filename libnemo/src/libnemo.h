#ifndef LIBNEMO_H
#define LIBNEMO_H

#include <string.h> // for size_t


//! \todo give nemo_prefix
typedef int status_t;

//! \todo give these NEMO-prefix
#define KERNEL_OK 0
#define KERNEL_INVOCATION_ERROR 1
#define KERNEL_ASSERTION_FAILURE 2
#define KERNEL_MEMORY_ERROR 3
#define KERNEL_UNKNOWN_ERROR 4


//-----------------------------------------------------------------------------
// RUNTIME DATA
//-----------------------------------------------------------------------------


/* Opaque pointer to network object */
typedef void* RTDATA;


//! \todo get rid of setReverse. Just deal with this in the host code
//! \todo dynamically resize the firing buffer?
/*!
 * \param setReverse
 * 		set reverse connectivity matrix, required e.g. for STDP
 * \param maxReadPeriod
 * 		maximum period (in cycles) between reads to the device firing buffer
 */
RTDATA
nemo_new_network(unsigned setReverse, unsigned maxReadPeriod);


// for debugging purposes it might be useful to fix the partition size
RTDATA
nemo_new_network_(
		unsigned setReverse,
		unsigned maxReadPeriod,
		unsigned maxPartitionSize);

void nemo_delete_network(RTDATA);


//-----------------------------------------------------------------------------
// NETWORK CONSTRUCTION
//-----------------------------------------------------------------------------


status_t
nemo_add_neuron(RTDATA,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);



status_t
nemo_add_synapses(RTDATA,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length);



/*! Read connectivity matrix back from device for a single neuron and delay. */
size_t
nemo_get_synapses(RTDATA rtdata,
		unsigned sourcePartition,
		unsigned sourceNeuron,
		unsigned delay,
		unsigned* targetPartition[],
		unsigned* targetNeuron[],
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
status_t
nemo_read_firing(RTDATA rtdata,
		unsigned** cycles,
		unsigned** neuronIdx,
		unsigned* nfired,
		unsigned* ncycles);


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
		unsigned preFireLen,
		unsigned postFireLen,
		float* preFireFn,
		float* postFireFn,
		float maxWeight,
		float minWeight);


//-----------------------------------------------------------------------------
// SIMULATION STEPPING
//-----------------------------------------------------------------------------


/*! Run simulation for an additional single cycle (1ms)
 *
 * Neurons can be optionally be forced to fire using the two arguments
 *
 * \param fstimCount
 * 		Length of fstimIdx
 * \param fstimIdx
 * 		Indices of the neurons which should be forced to fire this cycle.
 */
status_t
nemo_step(RTDATA rtdata, size_t fstimCount, unsigned fstimIdx[]);


status_t
nemo_apply_stdp(RTDATA rtdata, float stdpReward);


/* Force all allocated memory onto the device. Calling this is not required
 * during normal operation, as step invokes it on first call, but can be used
 * for testing */
status_t
nemo_start_simulation(RTDATA);


//-----------------------------------------------------------------------------
// ERROR HANDLING
//-----------------------------------------------------------------------------

/*! \return
 * 		string describing the most recent error (if any)
 */
const char*
nemo_strerror(RTDATA);



//-----------------------------------------------------------------------------
// DEVICE PROPERTIES
//-----------------------------------------------------------------------------

//! \return number of cuda-enabled devices of compute capability 1.0 or greater
int
nemo_device_count(void);


#endif
