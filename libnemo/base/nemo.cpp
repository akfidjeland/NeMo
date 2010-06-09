/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file nemo.cpp

/*! C API for libnemo
 *
 * This simply wrapes the API exposed in nemo::Simulation
 */

#include <assert.h>

extern "C" {
#include "nemo.h"
}

#include "Simulation.hpp"
#include "Configuration.hpp"
#include "Network.hpp"
#include "exception.hpp"
#include "nemo_error.h"
#include <nemo_config.h>

/* We cannot propagate exceptions via the C API, so we catch all and convert to
 * error codes instead. Error descriptions are stored on a per-process basis. */

static std::string g_lastError;
static nemo_status_t g_lastCallStatus = NEMO_OK;


void
setResult(const char* msg, nemo_status_t status) {
	g_lastError = msg;
	g_lastCallStatus = status;
}




/* Call method on wrapped object, and /set/ status and error */
#define CALL(call) {                                                          \
        g_lastCallStatus = NEMO_OK;                                           \
        try {                                                                 \
            call;                                                             \
        } catch (nemo::exception& e) {                                        \
            setResult(e.what(), e.errno());                                   \
        } catch (std::exception& e) {                                         \
            setResult(e.what(), NEMO_UNKNOWN_ERROR);                          \
        } catch (...) {                                                       \
            setResult("unknown exception", NEMO_UNKNOWN_ERROR);               \
        }                                                                     \
    }

/* Call method on wrapper object, and return status and error */
#define CATCH_(T, ptr, call) {                                                \
        nemo::T* obj = static_cast<nemo::T*>(ptr);                            \
        CALL(obj->call)                                                       \
        return g_lastCallStatus;                                              \
	}

/* Call method on wrapper object, set output value, and return status and error */
#define CATCH(T, ptr, call, ret) {                                            \
        nemo::T* obj = static_cast<nemo::T*>(ptr);                            \
        CALL(ret = obj->call);                                                \
        return g_lastCallStatus;                                              \
	}

#define NOCATCH(T, ptr, call) static_cast<nemo::T*>(ptr)->call



nemo_network_t
nemo_new_network()
{
	return static_cast<nemo_network_t>(new nemo::Network());
}


void
nemo_delete_network(nemo_network_t net)
{
	delete static_cast<nemo::Network*>(net);
}



nemo_configuration_t
nemo_new_configuration()
{
	return static_cast<nemo_configuration_t>(new nemo::Configuration());
}



void
nemo_delete_configuration(nemo_configuration_t conf)
{
	delete static_cast<nemo::Configuration*>(conf);
}



nemo_simulation_t
nemo_new_simulation(nemo_network_t net_ptr, nemo_configuration_t conf_ptr)
{
	nemo::Network* net = static_cast<nemo::Network*>(net_ptr);
	nemo::Configuration* conf = static_cast<nemo::Configuration*>(conf_ptr);
	try {
		return static_cast<nemo_simulation_t>(nemo::Simulation::create(*net, *conf));
	} catch(nemo::exception& e) {
		setResult(e.what(), e.errno());
		return NULL;
	} catch(std::exception& e) {
		setResult(e.what(), NEMO_UNKNOWN_ERROR);
		return NULL;
	}
}



void
nemo_delete_simulation(nemo_simulation_t sim)
{
	delete static_cast<nemo::Simulation*>(sim);
}



nemo_status_t
nemo_add_neuron(nemo_network_t net,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH_(Network, net, addNeuron(idx, a, b, c, d, u, v, sigma));
}



nemo_status_t
nemo_add_synapse(nemo_network_t net,
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char is_plastic)
{
	CATCH_(Network, net, addSynapse(source, target, delay, weight, is_plastic));
}



nemo_status_t
nemo_add_synapses(nemo_network_t net,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	CATCH_(Network, net, addSynapses(source, targets, delays, weights, is_plastic, length));
}



nemo_status_t
nemo_neuron_count(nemo_network_t net, unsigned* ncount)
{
	CATCH(Network, net, neuronCount(), *ncount);
}



nemo_status_t
nemo_get_synapses(nemo_simulation_t ptr,
		unsigned source,
		unsigned* targets_[],
		unsigned* delays_[],
		float* weights_[],
		unsigned char* plastic_[],
		size_t* len)
{
	const std::vector<unsigned>* targets;
	const std::vector<unsigned>* delays;
	const std::vector<float>* weights;
	const std::vector<unsigned char>* plastic;
	nemo::Simulation* sim = static_cast<nemo::Simulation*>(ptr);
	CALL(sim->getSynapses(source, &targets, &delays, &weights, &plastic));
	if(NEMO_OK == g_lastCallStatus) {
		*targets_ = const_cast<unsigned*>(&(*targets)[0]);
		*delays_ = const_cast<unsigned*>(&(*delays)[0]);
		*weights_ = const_cast<float*>(&(*weights)[0]);
		*plastic_ = const_cast<unsigned char*>(&(*plastic)[0]);
		*len = targets->size();
	}
	return g_lastCallStatus;
}



nemo_status_t
nemo_step(nemo_simulation_t sim, unsigned fstimIdx[], size_t fstimCount)
{
	CATCH_(Simulation, sim, step(std::vector<unsigned>(fstimIdx, fstimIdx + fstimCount)));
}



nemo_status_t
nemo_apply_stdp(nemo_simulation_t sim, float reward)
{
	CATCH_(Simulation, sim, applyStdp(reward));
}




nemo_status_t
nemo_read_firing(nemo_simulation_t ptr,
		unsigned* cycles_[],
		unsigned* nidx_[],
		unsigned* nfired,
		unsigned* ncycles)
{
	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* nidx;
	nemo::Simulation* sim = static_cast<nemo::Simulation*>(ptr);
	CALL(sim->readFiring(&cycles, &nidx));
	if(NEMO_OK == g_lastCallStatus) {
		*cycles_ = const_cast<unsigned*>(&(*cycles)[0]);
		*nidx_ = const_cast<unsigned*>(&(*nidx)[0]);
		*nfired = cycles->size();
		assert(cycles->size() == nidx->size());
	}
	return g_lastCallStatus;
}




nemo_status_t
nemo_flush_firing_buffer(nemo_simulation_t sim)
{
	CATCH_(Simulation, sim, flushFiringBuffer());
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


nemo_status_t
nemo_log_stdout(nemo_configuration_t conf)
{
	CATCH_(Configuration, conf, enableLogging());
}



nemo_status_t
nemo_elapsed_wallclock(nemo_simulation_t sim, unsigned long* elapsed)
{
	CATCH(Simulation, sim, elapsedWallclock(), *elapsed);
}



nemo_status_t
nemo_elapsed_simulation(nemo_simulation_t sim, unsigned long* elapsed)
{
	CATCH(Simulation, sim, elapsedSimulation(), *elapsed);
}



nemo_status_t
nemo_reset_timer(nemo_simulation_t sim)
{
	CATCH_(Simulation, sim, resetTimer());
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


nemo_status_t
nemo_set_stdp_function(nemo_configuration_t conf,
		float* pre_fn, size_t pre_len,
		float* post_fn, size_t post_len,
		float w_min,
		float w_max)
{
	CATCH_(Configuration, conf, setStdpFunction(
				std::vector<float>(pre_fn, pre_fn+pre_len),
				std::vector<float>(post_fn, post_fn+post_len),
				w_min, w_max));
}



nemo_status_t
nemo_set_cuda_firing_buffer_length(nemo_configuration_t conf, unsigned cycles)
{
	CATCH_(Configuration, conf, setCudaFiringBufferLength(cycles));
}



nemo_status_t
nemo_cuda_firing_buffer_length(nemo_configuration_t conf, unsigned* cycles)
{
	CATCH(Configuration, conf, cudaFiringBufferLength(), *cycles);
}



nemo_status_t
nemo_set_cuda_partition_size(nemo_configuration_t conf, unsigned size)
{
	CATCH_(Configuration, conf, setCudaPartitionSize(size));
}



nemo_status_t
nemo_set_cuda_device(nemo_configuration_t conf, int dev)
{
	CATCH_(Configuration, conf, setCudaDevice(dev));
}


const char*
nemo_strerror()
{
	return g_lastError.c_str();
}
