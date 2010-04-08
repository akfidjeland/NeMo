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
 * This simply wrapes the API exposed in nemo::Network */

extern "C" {
#include "nemo.h"
}

#include "CudaNetwork.hpp"
#include "nemo.hpp"
//! \todo combine these into a single header file
#include "DeviceAssertions.hpp"
#include "except.hpp"

/* We cannot propagate exceptions via the C API, so convert to an error code
 * instead */


/* Call method on network object, and /set/ status and error */
#define CATCH_(net, call) {                                                   \
        net->setStatus(NEMO_OK);                                              \
        try {                                                                 \
            call;                                                             \
        } catch (DeviceAllocationException& e) {                              \
            net->setErrorMsg(e.what());                                       \
            net->setStatus(NEMO_CUDA_MEMORY_ERROR);                           \
        } catch (KernelInvocationError& e) {                                  \
            net->setErrorMsg(e.what());                                       \
            net->setStatus(NEMO_CUDA_INVOCATION_ERROR);                       \
        } catch (DeviceAssertionFailure& e) {                                 \
            net->setErrorMsg(e.what());                                       \
            net->setStatus(NEMO_CUDA_ASSERTION_FAILURE);                      \
        } catch (std::exception& e) {                                         \
            net->setErrorMsg(e.what());                                       \
            net->setStatus(NEMO_UNKNOWN_ERROR);                               \
        } catch (...) {                                                       \
            net->setErrorMsg("unknown exception");                            \
            net->setStatus(NEMO_UNKNOWN_ERROR);                               \
        }                                                                     \
    }

/* Call method on network object, and /return/ status and error */
#define CATCH(ptr, call) {                                                    \
        Network* net = static_cast<Network*>(ptr);                            \
        CATCH_(net, net->m_impl->call)                                        \
        return net->status();                                                 \
	}


//! \todo enforce no throw in the class interface
/* Call function without handling exceptions */
#define NOCATCH(ptr, call) static_cast<Network*>(ptr)->m_impl->call



class Network
{
	public :

		Network(bool setReverse) :
			m_impl(nemo::Network::create(setReverse)),
			m_errorMsg("No error") { }

		//! \todo set partition size through a separate configuration function
		Network(bool setReverse, unsigned maxPartitionSize) :
			m_impl(new nemo::cuda::CudaNetwork(setReverse, maxPartitionSize)),
			m_errorMsg("No error") { }

		void setErrorMsg(const char* msg) { m_errorMsg = msg; }

		const char* lastErrorMsg() { return m_errorMsg.c_str(); }

		void setStatus(nemo_status_t s) { m_status = s; }

		nemo_status_t status() const { return m_status; }

		nemo::Network* m_impl;

	private :

		/* In addition to the runtime data, we need to keep track of the latest
		 * error and associated error message */
		std::string m_errorMsg;

		/* Status after last call */
		nemo_status_t m_status;

};


NETWORK
nemo_new_network(unsigned char usingStdp)
{
	return new Network((bool) usingStdp);
}



NETWORK
nemo_new_network_(unsigned char usingStdp, unsigned maxPartitionSize)
{
	return new Network((bool) usingStdp, maxPartitionSize);
}



void
nemo_delete_network(NETWORK mem)
{
	delete static_cast<Network*>(mem);
}



nemo_status_t
nemo_add_neuron(NETWORK rt,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH(rt, addNeuron(idx, a, b, c, d, u, v, sigma));
}



nemo_status_t
nemo_add_synapses(NETWORK network,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	CATCH(network, addSynapses(source,
				std::vector<unsigned>(targets, targets+length),
				std::vector<unsigned>(delays, delays+length),
				std::vector<float>(weights, weights+length),
				std::vector<unsigned char>(is_plastic, is_plastic+length)));
}



nemo_status_t
nemo_get_synapses(NETWORK ptr,
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
	Network* net = static_cast<Network*>(ptr);
	CATCH_(net, net->m_impl->getSynapses(source,
				&targets, &delays, &weights, &plastic));
	if(net->status() == NEMO_OK) {
		*targets_ = const_cast<unsigned*>(&(*targets)[0]);
		*delays_ = const_cast<unsigned*>(&(*delays)[0]);
		*weights_ = const_cast<float*>(&(*weights)[0]);
		*plastic_ = const_cast<unsigned char*>(&(*plastic)[0]);
		*len = targets->size();
	}
	return net->status();
}



nemo_status_t
nemo_init_simulation(NETWORK network)
{
	CATCH(network, initSimulation());
}



nemo_status_t
nemo_step(NETWORK network, unsigned fstimIdx[], size_t fstimCount)
{
	CATCH(network, stepSimulation(std::vector<unsigned>(fstimIdx, fstimIdx + fstimCount)));
}


nemo_status_t
nemo_apply_stdp(NETWORK network, float reward)
{
	CATCH(network, applyStdp(reward));
}




nemo_status_t
nemo_read_firing(NETWORK ptr,
		unsigned* cycles_[],
		unsigned* nidx_[],
		unsigned* nfired,
		unsigned* ncycles)
{
	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* nidx;
	Network* net = static_cast<Network*>(ptr);
	CATCH_(net, *ncycles = net->m_impl->readFiring(&cycles, &nidx));
	if(net->status() == NEMO_OK) {
		*cycles_ = const_cast<unsigned*>(&(*cycles)[0]);
		*nidx_ = const_cast<unsigned*>(&(*nidx)[0]);
		*nfired = cycles->size();
		assert(cycles->size() == nidx->size());
	}
	return net->status();
}



nemo_status_t
nemo_flush_firing_buffer(NETWORK network)
{
	NOCATCH(network, flushFiringBuffer());
	return NEMO_OK;
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


nemo_status_t
nemo_log_stdout(NETWORK network)
{
	CATCH(network, logToStdout())
}



unsigned long
nemo_elapsed_wallclock(NETWORK network)
{
	return NOCATCH(network, elapsedWallclock());
}



unsigned long
nemo_elapsed_simulation(NETWORK network)
{
	return NOCATCH(network, elapsedSimulation());
}


void
nemo_reset_timer(NETWORK network)
{
	NOCATCH(network, resetTimer());
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------


void
nemo_enable_stdp(NETWORK network,
		float* pre_fn,
		size_t pre_len,
		float* post_fn,
		size_t post_len,
		float w_min,
		float w_max)
{
	NOCATCH(network, enableStdp(
				std::vector<float>(pre_fn, pre_fn+pre_len),
				std::vector<float>(post_fn, post_fn+post_len),
				w_min, w_max));
}



nemo_status_t
nemo_set_firing_buffer_length(NETWORK network, unsigned cycles)
{
	CATCH(network, setFiringBufferLength(cycles));
}



nemo_status_t
nemo_get_firing_buffer_length(NETWORK ptr, unsigned* cycles)
{
	Network* net = static_cast<Network*>(ptr);                            \
	CATCH_(net, *cycles = net->m_impl->getFiringBufferLength());
	return net->status();
}



const char*
nemo_strerror(NETWORK network)
{
	return const_cast<char*>(static_cast<Network*>(network)->lastErrorMsg());
}

