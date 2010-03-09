#ifndef LIBNEMO_H
#define LIBNEMO_H

/*! \file libnemo.h
 *
 * \author Andreas Fidjeland
 * \date February 2010
 * \brief C API for the nemo spiking neural network simulator
 *
 * The C API wraps a C++ class which can be used directly.
 */
//! \todo write a brief overview
//! \todo mention CUDA
//! \todo briefly document the izhikevich model
//! \todo refer to own paper
//! \todo mention neuron indexing
//! \todo mention millisecond accuracy
//! \todo document STDP

#include <stddef.h> // for size_t

/*! Opaque pointer to network object. */
typedef void* NETWORK;

/*! Status of API calls which can fail. */
typedef int nemo_status_t;

/*! The call resulted in no errors */
#define NEMO_OK 0

/*! The CUDA driver reported an error */
#define NEMO_CUDA_INVOCATION_ERROR 1

/*! An assertion failed on the CUDA backend. Note that these assertions are not
 * enabled by default. Build library with -DDEVICE_ASSERTIONS to enable these */
#define NEMO_CUDA_ASSERTION_FAILURE 2

/*! A memory allocation failed on the CUDA device. */
#define NEMO_CUDA_MEMORY_ERROR 3
#define NEMO_UNKNOWN_ERROR 4


//-----------------------------------------------------------------------------
// RUNTIME DATA
//-----------------------------------------------------------------------------


/*! \name Initialisation
 * \{ */

//! \todo get rid of setReverse. Just deal with this in the host code
//! \todo dynamically resize the firing buffer?
/*! Create an empty network object
 *
 * \param setReverse
 * 		set reverse connectivity matrix, required e.g. for STDP
 * \param maxReadPeriod
 * 		maximum period (in cycles) between reads to the device firing buffer
 */
NETWORK
nemo_new_network(unsigned setReverse, unsigned maxReadPeriod);

/* \} */ // end init section


//-----------------------------------------------------------------------------
// CONFIGURATION
//-----------------------------------------------------------------------------

/*! \name Configuration */
/* \{ */ // begin configuration


/*! \copydoc nemo::Network::logToStdout */
nemo_status_t nemo_log_stdout(NETWORK);

/*! Enable spike-timing dependent plasticity in the simulation.
 *
 * \param prefire_fn
 * 		STDP function sampled at integer cycle intervals in the prefire part of
 * 		the STDP window
 * \param prefire_len
 * 		Length, in cycles, of the part of the STDP window that precedes the
 * 		postsynaptic firing.
 * \param postfire_fn
 * 		STDP function sampled at integer cycle intervals in the postfire part of
 * 		the STDP window
 * \param postfire_len
 * 		Length, in cycles, of the part of the STDP window that comes after the
 * 		postsynaptic firing.
 * \param max_weight
 * 		Weight beyond which excitatory synapses are not allowed to move
 * \param min_weight
 * 		Weight beyond which excitatory synapses are not allowed to move
 */
void
nemo_enable_stdp(NETWORK,
		float prefire_fn[],
		size_t prefire_len,
		float postfire_fn[],
		size_t postfire_len,
		float min_weight,
		float max_weight);

/* \} */ // end configuration



//-----------------------------------------------------------------------------
// NETWORK CONSTRUCTION
//-----------------------------------------------------------------------------

/*! \name Construction
 *
 * Networks are constructed by adding individual neurons, and single or groups
 * of synapses to the network. Neurons are given indices (from 0) which should
 * be unique for each neuron. When adding synapses the source or target neurons
 * need not necessarily exist yet, but should be defined before the network is
 * finalised.
 *
 * \{ */

//! \todo make sure we handle the issue of non-unique indices
//! \todo add description of neuron indices
/*! \copydoc nemo::Network::addNeuron */
nemo_status_t
nemo_add_neuron(NETWORK,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);


//! \todo add method to add a single synapse
//! \todo add documentation on the possible failure codes


/*! Add to the network a group of synapses with the same presynaptic neuron
 *
 * \param source
 * 		Index of source neuron
 * \param targets
 * 		Indices of target neurons
 * \param delays
 * 		Synapse conductance delays in milliseconds
 * \param weights
 * 		Synapse weights
 * \param is_plastic
 * 		Specifies for each synapse whether or not it is plastic. See section on STDP.
 * \param length
 * 		Number of synapses.
 *
 * \pre
 * 		Each of \a targets, \a delays, \a weights, and \a is_plastic contains
 * 		\a length elements.
 */
nemo_status_t
nemo_add_synapses(NETWORK,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length);

/* \} */ // end construction group



//-----------------------------------------------------------------------------
// SIMULATION STEPPING
//-----------------------------------------------------------------------------

/*! \name Simulation
 * \{ */

//! \todo rename to finalise network
/*! \copydoc nemo::Network::startSimulation */
nemo_status_t
nemo_start_simulation(NETWORK);

/*! Run simulation for a single cycle (1ms)
 *
 * Neurons can be optionally be forced to fire using the two arguments
 *
 * \param fstimIdx
 * 		Indices of the neurons which should be forced to fire this cycle.
 * \param fstimCount
 * 		Length of fstimIdx
 */
nemo_status_t
nemo_step(NETWORK, unsigned fstimIdx[], size_t fstimCount);


/*! \copydoc nemo::Network::applyStdp */
nemo_status_t
nemo_apply_stdp(NETWORK, float reward);


//-----------------------------------------------------------------------------
// FIRING PROBE
//-----------------------------------------------------------------------------

/*! \name Simulation (firing)
 * \{ */

/*! Return all buffered firing data and empty buffers.
 *
 * The three arrays together form a vector of 3-tuples specifying cycle,
 * partition index, and neuron index for all the fired neurons. 
 *
 * The last two output variables contain the number of neurons and the number of
 * cycles for which we have firing.
 */
nemo_status_t
nemo_read_firing(NETWORK,
		unsigned* cycles[],
		unsigned* neuronIdx[],
		unsigned* nfired,
		unsigned* ncycles);


/*! \copydoc nemo::Network::flushFiringBuffer */
nemo_status_t
nemo_flush_firing_buffer(NETWORK);

/* \} */ // end firing group


//-----------------------------------------------------------------------------
// QUERIES
//-----------------------------------------------------------------------------

/*! Read connectivity matrix back from device for a single neuron and delay. */
size_t
nemo_get_synapses(NETWORK,
		unsigned sourcePartition,
		unsigned sourceNeuron,
		unsigned delay,
		unsigned* targetPartition[],
		unsigned* targetNeuron[],
		float* weights[],
		unsigned char* plastic[]);




/* \} */ // end simulation group




//-----------------------------------------------------------------------------
// ERROR HANDLING
//-----------------------------------------------------------------------------

/*! \name Error handling
 * \{ */

//! \todo consider putting the error codes here

/*! \return
 * 		string describing the most recent error (if any)
 */
const char*
nemo_strerror(NETWORK);

/*! \} */  //end error group




/*! \name Finalization
 * \{ */

/*! Delete network object, freeing up all its associated resources */
void nemo_delete_network(NETWORK);

/* \} */ // end finalize section



//-----------------------------------------------------------------------------
// TIMERS
//-----------------------------------------------------------------------------

/*! \name Simulation (timing) \{ */

/*! \copydoc nemo::Network::elapsedWallclock */
unsigned long nemo_elapsed_wallclock(NETWORK);

/*! \copydoc nemo::Network::elapsedSimulation */
unsigned long nemo_elapsed_simulation(NETWORK);

/*! \copydoc nemo::Network::resetTimer */
void nemo_reset_timer(NETWORK);

/* \} */ // end timing section


//-----------------------------------------------------------------------------
// DEBUGGING/INTERNALS
//-----------------------------------------------------------------------------

// for debugging purposes it might be useful to fix the partition size
NETWORK
nemo_new_network_(
		unsigned setReverse,
		unsigned maxReadPeriod,
		unsigned maxPartitionSize);



#endif
