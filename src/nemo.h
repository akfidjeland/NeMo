#ifndef NEMO_H
#define NEMO_H

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file nemo.h
 *
 * \brief C API for the NeMo spiking neural network simulator
 */

/*! \mainpage NeMo C++/C API reference
 *
 * NeMo is a C++ class library, with interfaces for C, Python, and Matlab.
 *
 *
 * The C++ API is based around three classes: \a nemo::Configuration, \a
 * nemo::Network, and \a nemo::Simulation. The C API gives access to these via
 * opaque pointers, which are explicitly managed by the user. The C API
 * functions are documented in \a nemo.h.
 *
 * The Python and Matlab APIs are documented elsewhere.
 */ 

#ifdef __cplusplus
extern "C" {
#endif


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


NEMO_DLL_PUBLIC
void nemo_delete_configuration(nemo_configuration_t);


/*! \copydoc nemo::Configuration::enableLogging */
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


/*! \copydoc nemo::Configuration::backendDescription */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_backend(nemo_configuration_t conf, backend_t* backend);


/*! \copydoc nemo::Configuration::backendDescription */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_backend_description(nemo_configuration_t conf, const char** descr);


/*! \copydoc nemo::Configuration::setWriteOnlySynapses */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_write_only_synapses(nemo_configuration_t conf);



/* \} */ // end configuration



//-----------------------------------------------------------------------------
// NETWORK CONSTRUCTION
//-----------------------------------------------------------------------------

/*! \name Construction
 *
 * Networks are constructed by adding individual neurons and of synapses to the
 * network. Neurons are given indices (ideally, but not necessarily starting
 * from 0) which should be unique for each neuron. When adding synapses the
 * source or target neurons need not necessarily exist yet, but need to be
 * defined before the simulation is created.
 *
 * \{ */


/*! Create an empty network object */
NEMO_DLL_PUBLIC
nemo_network_t nemo_new_network();


/*! Delete network object, freeing up all its associated resources */
NEMO_DLL_PUBLIC
void nemo_delete_network(nemo_network_t);


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


/*! Create a new simulation from an existing populated network and a
 * configuration */
NEMO_DLL_PUBLIC
nemo_simulation_t nemo_new_simulation(nemo_network_t, nemo_configuration_t);


/*! Delete simulation object, freeing up all its associated resources */
NEMO_DLL_PUBLIC
void nemo_delete_simulation(nemo_simulation_t);


/*! Run simulation for a single cycle (1ms)
 *
 * Neurons can optionally be forced to fire using \a fstim_nidx and \a
 * fstim_count.  Input current can be provided to a set of neurons using \a
 * istim_nidx, \a istim_current, and \a istim_count.
 *
 * \param fstim_nidx
 * 		Indices of the neurons which should be forced to fire this cycle.
 * \param fstim_count
 * 		Length of \a fstim_nidx
 * \param istim_nidx
 * 		Indices of neurons which should receive external current stimulus this
 * 		cycle.
 * \param istim_current
 * 		The corresponding vector of current
 * \param istim_count
 * 		Length of \a istim_nidx \b and \a istim_current
 * \param[out] fired
 * 		Vector which fill be filled with the indices of the neurons which fired
 * 		this cycle. Set to NULL if the firing output is ignored.
 * \param[out] fired_count
 * 		Number of neurons which fired this cycle, i.e. the length of \a fired.
 * 		Set to NULL if the firing output is ignored.
 *
 * \return
 * 		NEMO_OK if operation succeeded, some other value otherwise.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_step(nemo_simulation_t,
		unsigned fstim_nidx[], size_t fstim_count,
		unsigned istim_nidx[], float istim_current[], size_t istim_count,
		unsigned* fired[], size_t* fired_count);


/*! \copydoc nemo::Simulation::applyStdp */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_apply_stdp(nemo_simulation_t, float reward);


//-----------------------------------------------------------------------------
// QUERIES
//-----------------------------------------------------------------------------

/*! \name Querying the network
 *
 * Neuron parameters (static) and state varaibles (dynamic) may be read read
 * back either during construction or simulation. The same function names are
 * used in both cases, but functions are postfixed with '_n' or '_s' to denote
 * network or simulation functions.
 *
 * The synapse state can also be read back during simulation. Synapses are
 * referred to via a synapse_id (see \a nemo_add_synapse). The weights may
 * change at run-time, while the other synapse data is static.
 * \{ */

NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_membrane_potential(nemo_simulation_t sim, unsigned neuron, float* v);



/*! Get a single state variable for a single neuron during construction
 *
 * \param[in] net network object
 * \param[in] neuron neuron index
 * \param[in] var state variable index
 * \param[out] val value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised. \a val is undefined unless the return value is NEMO_OK.
 *
 * For the Izhikevich model the variable indices are 0 = u, 1 = v.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_neuron_state_n(nemo_network_t net, unsigned neuron, unsigned var, float* val);



/*! Get a single parameter for a single neuron during simulation
 *
 * \param[in] net network object
 * \param[in] neuron neuron index
 * \param[in] param parameter index
 * \param[out] val value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or parameter indices are invalid. Other errors may also be
 * 		raised. \a val is undefined unless the return value is NEMO_OK.
 *
 * For the Izhikevich model the parameter indices are 0 = a, 1 = b, 2 = c, 3 = d.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_neuron_parameter_n(nemo_network_t net, unsigned neuron, unsigned param, float* val);



/*! Get a single state variable for a single neuron during simulation
 *
 * \param[in] sim simulation object
 * \param[in] neuron neuron index
 * \param[in] var state variable index
 * \param[out] val value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised. \a val is undefined unless the return value is NEMO_OK.
 *
 * For the Izhikevich model the variable indices are 0 = u, 1 = v.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_neuron_state_s(nemo_simulation_t sim, unsigned neuron, unsigned var, float* val);



/*! Get a single parameter for a single neuron during simulation
 *
 * \param[in] sim simulation object
 * \param[in] neuron neuron index
 * \param[in] param parameter index
 * \param[out] val value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or parameter indices are invalid. Other errors may also be
 * 		raised. \a val is undefined unless the return value is NEMO_OK.
 *
 * For the Izhikevich model the parameter indices are 0 = a, 1 = b, 2 = c, 3 = d.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_neuron_parameter_s(nemo_simulation_t sim, unsigned neuron, unsigned param, float* val);


/*! Get synapse target for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param[out] targets
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_targets(nemo_simulation_t, synapse_id synapses[], size_t len, unsigned* targets[]);


/*! Get conductance delays for the specified synapses
 *
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param[out] delays
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_delays(nemo_simulation_t, synapse_id synapses[], size_t len, unsigned* delays[]);


/*! Get synapse ids for synapses with the given source id
 *
 * \param source source neuron id
 * \param[out] synapses array of synapse ids
 * \param[out] len length of \a synapses array
 *
 * The output array is only valid until the next call to
 * \a nemo_get_synapses_from
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_synapses_from(nemo_simulation_t, unsigned source, synapse_id *synapses[], size_t* len);


/*! Get weights for the specified synapses
 *
 * \param ptr
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param[out] weights
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_weights(nemo_simulation_t ptr, synapse_id synapses[], size_t len, float* weights[]);


/*! Get boolean plasticity status for the specified synapses
 *
 * \param ptr
 * \param synapses list of synapse ids (\see nemo_add_synapse)
 * \param len length of \a synapses
 * \param[out] plastic
 * 		vector of length \a len to be set with synapse state. The memory is
 * 		managed by the simulation object and is valid until the next call to
 * 		this function.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_get_plastic(nemo_simulation_t ptr, synapse_id synapses[], size_t len, unsigned char* plastic[]);


/* \} */ // end simulation group


//-----------------------------------------------------------------------------
// MODIFYING THE NETWROK
//-----------------------------------------------------------------------------

/*! \name Modifying the network
 *
 * Neuron parameters and state variables can be modified during both
 * construction and simulation. The same function names are used in both cases,
 * but functions are postfixed with '_n' or '_s' to denote network or
 * simulation functions.
 *
 * In the current version of NeMo synapses can not be modified during
 * simulation.
 * \{ */

/*! Modify the parameters/state for a single neuron during construction
 *
 * The neuron must already exist.
 *
 * \see nemo_add_neuron for parameters
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_n(nemo_network_t net,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);



/*! Modify the parameters/state for a single neuron during simulation
 *
 * The neuron must already exist.
 *
 * \see nemo_add_neuron for parameters
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_s(nemo_simulation_t sim,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma);



/*! Modify a single state variable for a single neuron during construction
 *
 * \param[in] net network object
 * \param[in] neuron neuron index
 * \param[in] var state variable index
 * \param[in] val new value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised.
 *
 * For the Izhikevich model the variable indices are 0 = u, 1 = v.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_state_n(nemo_network_t net, unsigned neuron, unsigned var, float val);



/*! Modify a single parameter for a single neuron during construction
 *
 * \param[in] net network object
 * \param[in] neuron neuron index
 * \param[in] param parameter index
 * \param[in] val new value of the parameter
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised.
 *
 * For the Izhikevich model the parameter indices are 0 = a, 1 = b, 2 = c, 3 = d.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_parameter_n(nemo_network_t net, unsigned neuron, unsigned param, float val);



/*! Modify a single state variable for a single neuron during simulation
 *
 * \param[in] sim simulation object
 * \param[in] neuron neuron index
 * \param[in] var state variable index
 * \param[in] val new value of the state variable
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised. \a val is undefined unless the return value is NEMO_OK.
 *
 * For the Izhikevich model the variable indices are 0 = u, 1 = v.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_state_s(nemo_simulation_t sim, unsigned neuron, unsigned var, float val);



/*! Modify a single parameter for a single neuron during simulation
 *
 * \param[in] sim simulation object
 * \param[in] neuron neuron index
 * \param[in] param parameter index
 * \param[in] val new value of the parameter
 *
 * \return NEMO_OK if no errors occurred. Returns NEMO_INVALID_INPUT if either
 * 		the neuron or state variable indices are invalid. Other errors may also
 * 		be raised.
 *
 * For the Izhikevich model the parameter indices are 0 = a, 1 = b, 2 = c, 3 = d.
 */
NEMO_DLL_PUBLIC
nemo_status_t
nemo_set_neuron_parameter_s(nemo_simulation_t sim, unsigned neuron, unsigned param, float val);


/* \} */ // end modification group


//-----------------------------------------------------------------------------
// TIMERS
//-----------------------------------------------------------------------------


/*! \name Simulation timing
 *
 * The simulation has two internal timers which keep track of the elapsed \e
 * simulated time and \e wallclock time. Both timers measure from the first
 * simulation step, or from the last timer reset, whichever comes last.
 *
 * \{ */

//! \todo change to using output arguments and return status instead.

/*! \copydoc nemo::Simulation::elapsedWallclock */
NEMO_DLL_PUBLIC
nemo_status_t nemo_elapsed_wallclock(nemo_simulation_t, unsigned long*);

/*! \copydoc nemo::Simulation::elapsedSimulation */
NEMO_DLL_PUBLIC
nemo_status_t nemo_elapsed_simulation(nemo_simulation_t, unsigned long*);

/*! \copydoc nemo::Simulation::resetTimer */
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




#ifdef __cplusplus
}
#endif

#endif