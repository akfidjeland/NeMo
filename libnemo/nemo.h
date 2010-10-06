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
#include <nemo/types.h>

/*! Only opaque pointers are exposed in the C API */
typedef void* nemo_network_t;
typedef void* nemo_simulation_t;
typedef void* nemo_configuration_t;

/*! Status of API calls which can fail. */
typedef int nemo_status_t;


NEMO_DLL_PUBLIC
const char* nemo_version();


//-----------------------------------------------------------------------------
// HARDWARE CONFIGURATION
//-----------------------------------------------------------------------------


/*! \return number of CUDA devices on this system.
 *
 * In case of error sets device count to 0 and return an error code. The
 * associated error message can read using nemo_strerror. Errors can be the
 * result of missing CUDA libraries, which from the users point of view may or
 * may not be considered an error */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_cuda_device_count(unsigned* count);


NEMO_DLL_PUBLIC
nemo_status_t
nemo_cuda_device_description(unsigned device, const char**);


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
 * \param min_weight
 * 		Weight beyond which inhibitory synapses are not allowed to move
 * \param max_weight
 * 		Weight beyond which excitatory synapses are not allowed to move
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_stdp_function(nemo_configuration_t,
		float prefire_fn[], size_t prefire_len,
		float postfire_fn[], size_t postfire_len,
		float min_weight,
		float max_weight);


/*! \copydoc nemo::Configuration::setCpuBackend */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_cpu_backend(nemo_configuration_t, int thread_count);


/*! \copydoc nemo::Configuration::cpuThreadCount */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_cpu_thread_count(nemo_configuration_t conf, int* thread_count);


/*! \copydoc nemo::Configuration::setCudaBackend */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_cuda_backend(nemo_configuration_t conf, int dev);


NEMO_DLL_PUBLIC
nemo_status_t
nemo_cuda_device(nemo_configuration_t conf, int* dev);


/*! \copydoc nemo::Configuration::backend */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_backend(nemo_configuration_t conf, backend_t* backend);


/*! \copydoc nemo::Configuration::backendDescription */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_backend_description(nemo_configuration_t conf, const char** descr);



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


/* Add a single synapse to network
 *
 * \a id
 * 		Unique id of this synapse (which can be used for run-time queries). Set
 * 		to NULL if this is not required.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_add_synapse(nemo_network_t,
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char is_plastic,
		synapse_id* id);



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
nemo_simulation_t nemo_new_simulation(nemo_network_t, nemo_configuration_t);


/*! Run simulation for a single cycle (1ms)
 *
 * Neurons can be optionally be forced to fire using the two arguments
 *
 * \param firing_stimulus
 * 		Indices of the neurons which should be forced to fire this cycle.
 * \param firing_stimulus_count
 * 		Length of \a firing_stimulus
 * \param fired (output)
 * 		Vector which fill be filled with the indices of the neurons which fired
 * 		this cycle. Set to NULL if the firing output is ignored.
 * \param fired_count (output)
 * 		Number of neurons which fired this cycle, i.e. the length of \a fired.
 * 		Set to NULL if the firing output is ignored.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_step(nemo_simulation_t,
		unsigned firing_stimulus[], size_t firing_stimulus_count,
		unsigned* fired[], size_t* fired_count);


/*! \copydoc nemo::Simulation::applyStdp */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_apply_stdp(nemo_simulation_t, float reward);


//-----------------------------------------------------------------------------
// QUERIES
//-----------------------------------------------------------------------------

/*! \name Simulation (queries)
 *
 * The synapse state can be read back at run-time by specifiying a list of
 * synpase ids (\see addSynapse). The weights may change at run-time, while the
 * other synapse data is static.
 * \{ */


/*! Get synapse target for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param targets (output)
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_targets(nemo_simulation_t ptr, synapse_id synapses[], size_t len, unsigned* targets[]);


/*! Get conductance delays for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param delays (output)
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_delays(nemo_simulation_t ptr, synapse_id synapses[], size_t len, unsigned* delays[]);


/*! Get weights for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param weights (output)
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_weights(nemo_simulation_t ptr, synapse_id synapses[], size_t len, float* weights[]);


/*! Get boolean plasticity status for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param weights (output)
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_plastic(nemo_simulation_t ptr, synapse_id synapses[], size_t len, unsigned char* plastic[]);


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
