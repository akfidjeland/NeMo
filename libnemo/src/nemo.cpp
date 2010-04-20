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
 * This simply wrapes the API exposed in nemo::Simulation */

extern "C" {
#include "nemo.h"
}

#include "CudaNetwork.hpp"
#include "Configuration.hpp"
#include "Network.hpp"
#include "DeviceAssertions.hpp"
#include "except.hpp"

/* We cannot propagate exceptions via the C API, so we catch all and convert to
 * an error codes instead */


/* Call method on wrapped object, and /set/ status and error */
#define CALL(ptr, call) {                                                     \
        ptr->setStatus(NEMO_OK);                                              \
        try {                                                                 \
            call;                                                             \
        } catch (DeviceAllocationException& e) {                              \
            ptr->setResult(e.what(), NEMO_CUDA_MEMORY_ERROR);                 \
        } catch (KernelInvocationError& e) {                                  \
            ptr->setResult(e.what(), NEMO_CUDA_INVOCATION_ERROR);             \
        } catch (DeviceAssertionFailure& e) {                                 \
            ptr->setResult(e.what(), NEMO_CUDA_ASSERTION_FAILURE);            \
        } catch (std::exception& e) {                                         \
            ptr->setResult(e.what(), NEMO_UNKNOWN_ERROR);                     \
        } catch (...) {                                                       \
			ptr->setResult("unknown exception", NEMO_UNKNOWN_ERROR);          \
        }                                                                     \
    }

/* Call method on wrapper object, and return status and error */
#define CATCH_(T, ptr, call) {                                                \
        Wrapper<nemo::T>* wrapper = static_cast<Wrapper<nemo::T>*>(ptr);      \
        CALL(wrapper, wrapper->data->call)                                    \
        return wrapper->status();                                             \
	}

/* Call method on wrapper object, set output value, and return status and error */
#define CATCH(T, ptr, call, ret) {                                            \
        Wrapper<nemo::T>* wrapper = static_cast<Wrapper<nemo::T>*>(ptr);      \
        CALL(wrapper, ret = wrapper->data->call);                             \
        return wrapper->status();                                             \
	}

#define NOCATCH(T, ptr, call) static_cast<Wrapper<nemo::T>*>(ptr)->data->call



class Catching
{
	public :

		Catching() : m_errorMsg("No error") { }

		void setResult(const char* msg, nemo_status_t status) {
			m_errorMsg = msg;
			m_status = status;
		}

		void setStatus(nemo_status_t s) { m_status = s; }

		//void setErrorMsg(const char* msg) { m_errorMsg = msg; }
		const char* errorMsg() const { return m_errorMsg.c_str(); }

		//void setStatus(nemo_status_t s) { m_status = s; }
		nemo_status_t status() const { return m_status; }

	private :

		/* In addition to the runtime data, we need to keep track of the latest
		 * error and associated error message */
		std::string m_errorMsg;

		/* Status after last call */
		nemo_status_t m_status;
};



template<class T>
class Wrapper : public Catching
{
	public :
		Wrapper() : data(new T()) {}
		Wrapper(T* data) : data(data) {}
		~Wrapper() { delete data; }
		T* data;
};


template<class T>
Wrapper<T>*
fromOpaque(void *ptr)
{
	return static_cast<Wrapper<T>*>(ptr);
}



nemo_network_t
nemo_new_network()
{
	return new Wrapper<nemo::Network>();
}


void
nemo_delete_network(nemo_network_t net)
{
	delete fromOpaque<nemo::Network>(net);
}



nemo_configuration_t
nemo_new_configuration()
{
	return new Wrapper<nemo::Configuration>();
}



void
nemo_delete_configuration(nemo_configuration_t conf)
{
	delete fromOpaque<nemo::Configuration>(conf);
}



nemo_simulation_t
nemo_new_simulation(nemo_network_t net_ptr, nemo_configuration_t conf_ptr)
{
	nemo::Network& net = *(fromOpaque<nemo::Network>(net_ptr)->data);
	nemo::Configuration& conf = *(fromOpaque<nemo::Configuration>(conf_ptr)->data);
	return new Wrapper<nemo::Simulation>(nemo::Simulation::create(net, conf));
}



void
nemo_delete_simulation(nemo_simulation_t sim)
{
	delete fromOpaque<nemo::Simulation>(sim);
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
nemo_add_synapses(nemo_network_t net,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	CATCH_(Network, net, addSynapses(source,
				std::vector<unsigned>(targets, targets+length),
				std::vector<unsigned>(delays, delays+length),
				std::vector<float>(weights, weights+length),
				std::vector<unsigned char>(is_plastic, is_plastic+length)));
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
	Wrapper<nemo::Simulation>* sim = fromOpaque<nemo::Simulation>(ptr);
	CALL(sim, sim->data->getSynapses(source,
				&targets, &delays, &weights, &plastic));
	if(sim->status() == NEMO_OK) {
		*targets_ = const_cast<unsigned*>(&(*targets)[0]);
		*delays_ = const_cast<unsigned*>(&(*delays)[0]);
		*weights_ = const_cast<float*>(&(*weights)[0]);
		*plastic_ = const_cast<unsigned char*>(&(*plastic)[0]);
		*len = targets->size();
	}
	return sim->status();
}



nemo_status_t
nemo_step(nemo_simulation_t sim, unsigned fstimIdx[], size_t fstimCount)
{
	CATCH_(Simulation, sim, stepSimulation(std::vector<unsigned>(fstimIdx, fstimIdx + fstimCount)));
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
	Wrapper<nemo::Simulation>* sim = fromOpaque<nemo::Simulation>(ptr);
	CALL(sim, sim->data->readFiring(&cycles, &nidx));
	if(sim->status() == NEMO_OK) {
		*cycles_ = const_cast<unsigned*>(&(*cycles)[0]);
		*nidx_ = const_cast<unsigned*>(&(*nidx)[0]);
		*nfired = cycles->size();
		assert(cycles->size() == nidx->size());
	}
	return sim->status();
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



//! \todo set status here as well, return data via pointer
unsigned long
nemo_elapsed_wallclock(nemo_simulation_t sim)
{
	return NOCATCH(Simulation, sim, elapsedWallclock());
}



unsigned long
nemo_elapsed_simulation(nemo_simulation_t sim)
{
	return NOCATCH(Simulation, sim, elapsedWallclock());
}



void
nemo_reset_timer(nemo_simulation_t sim)
{
	NOCATCH(Simulation, sim, resetTimer());
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


nemo_status_t
nemo_enable_stdp(nemo_configuration_t conf,
		float* pre_fn,
		size_t pre_len,
		float* post_fn,
		size_t post_len,
		float w_min,
		float w_max)
{
	CATCH_(Configuration, conf, setStdpFunction(
				std::vector<float>(pre_fn, pre_fn+pre_len),
				std::vector<float>(post_fn, post_fn+post_len),
				w_min, w_max));
}



nemo_status_t
nemo_set_firing_buffer_length(nemo_configuration_t conf, unsigned cycles)
{
	CATCH_(Configuration, conf, setCudaFiringBufferLength(cycles));
}



nemo_status_t
nemo_get_firing_buffer_length(nemo_configuration_t conf, unsigned* cycles)
{
	CATCH(Configuration, conf, cudaFiringBufferLength(), *cycles);
}



nemo_status_t
nemo_set_cuda_partition_size(nemo_configuration_t conf, unsigned size)
{
	CATCH_(Configuration, conf, setCudaPartitionSize(size));
}



const char*
nemo_strerror(void* ptr)
{
	return static_cast<Catching*>(ptr)->errorMsg();
}
