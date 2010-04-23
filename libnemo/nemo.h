#ifndef NEMO_H
#define NEMO_H

#ifdef __cplusplus
extern "C" {
#endif

/*! \file nemo.h
 *
 * \brief C API for the nemo spiking neural network simulator
 */

#include <stddef.h> // for size_t
#include <nemo/config.h>
#include <nemo/constants.h>

/*! Only opaque pointers are exposed in the C API */
typedef void* nemo_network_t;
typedef void* nemo_simulation_t;
typedef void* nemo_configuration_t;

/*! Status of API calls which can fail. */
typedef int nemo_status_t;


//-----------------------------------------------------------------------------
// CONFIGURATION
//-----------------------------------------------------------------------------

/*! \name Configuration */
/* \{ */ // begin configuration

NEMO_DLL_PUBLIC
nemo_configuration_t nemo_new_configuration();

/*! \copydoc nemo::Network::logToStdout */
NEMO_DLL_PUBLIC
nemo_status_t nemo_log_stdout(nemo_configuration_t);

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
 * 		Weight beyond which inhibitory synapses are not allowed to move
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_stdp_function(nemo_configuration_t,
		float prefire_fn[], size_t prefire_len,
		float postfire_fn[], size_t postfire_len,
		float min_weight,
		float max_weight);




/*! \copydoc nemo::Configuration::setCudaFiringBufferLength */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_cuda_firing_buffer_length(nemo_configuration_t, unsigned cycles);


/*! \copydoc nemo::Configuration::cudaFiringBufferLength */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_cuda_firing_buffer_length(nemo_configuration_t, unsigned* cycles);


/*! \copydoc nemo::Configuration::setCudaPartitionSize */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_cuda_partition_size(nemo_configuration_t conf, unsigned size);


NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_cuda_device(nemo_configuration_t conf, int dev);


/*! \copydoc nemo::Configuration::setFractionalBits */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_fractional_bits(nemo_configuration_t conf, unsigned bits);


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


/*! Create an empty network object */
NEMO_DLL_PUBLIC
nemo_network_t nemo_new_network();

//! \todo make sure we handle the issue of non-unique indices
//! \todo add description of neuron indices
/*! \copydoc nemo::Network::addNeuron */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_add_neuron(nemo_network_t,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);


//! \todo add method to add a single synapse
//! \todo add documentation on the possible failure codes


/* Add a single synapse to network */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_add_synapse(nemo_network_t,
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char is_plastic);


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
NEMO_DLL_PUBLIC
nemo_status_t
nemo_add_synapses(nemo_network_t,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length);


NEMO_DLL_PUBLIC
nemo_status_t
nemo_neuron_count(nemo_network_t net, unsigned* ncount);


/* \} */ // end construction group



//-----------------------------------------------------------------------------
// SIMULATION
//-----------------------------------------------------------------------------

/*! \name Simulation
 * \{ */

NEMO_DLL_PUBLIC
nemo_simulation_t nemo_new_simulation(nemo_network_t, nemo_simulation_t);

/*! \copydoc nemo::Network::initSimulation */
//nemo_status_t
//nemo_init_simulation(nemo_network_t);

/*! Run simulation for a single cycle (1ms)
 *
 * Neurons can be optionally be forced to fire using the two arguments
 *
 * \param fstimIdx
 * 		Indices of the neurons which should be forced to fire this cycle.
 * \param fstimCount
 * 		Length of fstimIdx
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_step(nemo_simulation_t, unsigned fstimIdx[], size_t fstimCount);


/*! \copydoc nemo::Simulation::applyStdp */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_apply_stdp(nemo_simulation_t, float reward);


//-----------------------------------------------------------------------------
// FIRING PROBE
//-----------------------------------------------------------------------------

/*! \name Simulation (firing)
 *
 * The indices of the fired neurons are buffered on the device, and can be read
 * back at run-time. The desired size of the buffer is specified when
 * constructing the network. Each read empties the buffer. To avoid overflow if
 * the firing data is not needed, call \ref nemo_flush_firing_buffer periodically.
 *
 * \{ */

/*! Return contents of firing buffer in the output parameters.
 *
 * \param[out] cycles
 * 		Cycle numbers (relative to start of buffer) at which neurons fired
 * \param[out] nidx
 * 		Neuron indices of fired neurons
 * \param[out] nfired
 * 		Number of neurons which fired since the previous call
 * 		to \ref nemo_read_firing
 * \param[out] ncycles
 * 		Number of cycles for which firing data is returned
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_read_firing(nemo_simulation_t,
		unsigned* cycles[],
		unsigned* nidx[],
		unsigned* nfired,
		unsigned* ncycles);


/*! \copydoc nemo::Network::flushFiringBuffer */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_flush_firing_buffer(nemo_simulation_t);

/* \} */ // end firing group


//-----------------------------------------------------------------------------
// QUERIES
//-----------------------------------------------------------------------------

/*! \name Simulation (queries)
 *
 * If STDP is enabled, the synaptic weights may change at
 * run-time. The user can read these back on a per-(source)
 * neuron basis.
 * \{ */

/*! Read connectivity matrix back from device for a single neuron and delay.
 * Every call to this function will return synapses in the same order.
 * The output vectors are valid until the next call to this function.
 *
 * \post
 * 		Output vectors \a targetNeuron, \a weights, \a delays,
 * 		and \a is_plastic all have length \a len
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_synapses(nemo_simulation_t,
		unsigned sourceNeuron,
		unsigned* targetNeuron[],
		unsigned* delays[],
		float* weights[],
		unsigned char* is_plastic[],
		size_t* len);



/* \} */ // end simulation group


//-----------------------------------------------------------------------------
// TIMERS
//-----------------------------------------------------------------------------


/*! \name Simulation (timing)
 *
 * The simulation has two internal timers which keep track of the elapsed \e
 * simulated time and \e wallclock time. Both timers measure from the first
 * simulation step, or from the last timer reset, whichever comes last.
 *
 * \{ */

//! \todo change to using output arguments and return status instead.

/*! \copydoc nemo::Network::elapsedWallclock */
NEMO_DLL_PUBLIC
nemo_status_t nemo_elapsed_wallclock(nemo_simulation_t, unsigned long*);

/*! \copydoc nemo::Network::elapsedSimulation */
NEMO_DLL_PUBLIC
nemo_status_t nemo_elapsed_simulation(nemo_simulation_t, unsigned long*);

/*! \copydoc nemo::Network::resetTimer */
NEMO_DLL_PUBLIC
nemo_status_t nemo_reset_timer(nemo_simulation_t);

/* \} */ // end timing section




//-----------------------------------------------------------------------------
// ERROR HANDLING
//-----------------------------------------------------------------------------

/*! \name Error handling
 *
 * The API functions generally return an error status of type \ref nemo_status_t.
 * A non-zero value indicates an error. An error string describing this error
 * is stored internally and can be queried by the user.
 *
 * \{ */

//! \todo consider putting the error codes here

/*! \return
 * 		string describing the most recent error (if any)
 */
NEMO_DLL_PUBLIC
const char* nemo_strerror();

/*! \} */  //end error group



/*! \name Finalization
 * \{ */


/*! Delete network object, freeing up all its associated resources */
NEMO_DLL_PUBLIC
void nemo_delete_network(nemo_network_t);


NEMO_DLL_PUBLIC
void nemo_delete_configuration(nemo_configuration_t);


/*! Delete simulation object, freeing up all its associated resources */
NEMO_DLL_PUBLIC
void nemo_delete_simulation(nemo_simulation_t);

/* \} */ // end finalize section

#ifdef __cplusplus
}
#endif

#endif
