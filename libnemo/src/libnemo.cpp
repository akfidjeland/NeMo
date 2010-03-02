extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
#include "DeviceAssertions.hpp"
#include "except.hpp"

/* We cannot propagate exceptions via the C API, so convert to an error code
 * instead */


/* Call method on network object, and /set/ status and error */
#define CATCH_(net, call) {                                                   \
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
        net->setStatus(NEMO_OK);                                              \
    }

/* Call method on network object, and /return/ status and error */
#define CATCH(ptr, call) {                                                    \
        Network* net = static_cast<Network*>(ptr);                            \
        CATCH_(net, net->call)                                                \
        return net->status();                                                 \
	}


//! \todo enforce no throw in the class interface
/* Call function without handling exceptions */
#define NOCATCH(ptr, call) static_cast<Network*>(ptr)->call


class Network : public nemo::RuntimeData {

	public :

		Network(bool setReverse, unsigned maxReadPeriod) :
			RuntimeData(setReverse, maxReadPeriod),
			m_errorMsg("No error") { } ;

		Network(bool setReverse,
				unsigned maxReadPeriod,
				unsigned maxPartitionSize) :
			RuntimeData(setReverse, maxReadPeriod, maxPartitionSize),
			m_errorMsg("No error") { } ;

		void setErrorMsg(const char* msg) { m_errorMsg = msg; }

		const char* lastErrorMsg() { return m_errorMsg.c_str(); }

		void setStatus(nemo_status_t s) { m_status = s; }

		nemo_status_t status() const { return m_status; }

	private :

		/* In addition to the runtime data, we need to keep track of the latest
		 * error and associated error message */
		std::string m_errorMsg;

		/* Status after last call */
		nemo_status_t m_status;
};


NETWORK
nemo_new_network(unsigned setReverse, unsigned maxReadPeriod)
{
	return new Network((bool) setReverse, maxReadPeriod);
}



NETWORK
nemo_new_network_(
		unsigned setReverse,
		unsigned maxReadPeriod,
		unsigned maxPartitionSize)
{
	return new Network((bool) setReverse, maxReadPeriod, maxPartitionSize);
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
	CATCH(network, addSynapses(source, targets, delays, weights, is_plastic, length));
}



size_t
nemo_get_synapses(NETWORK /*network*/,
		unsigned /*sourcePartition*/,
		unsigned /*sourceNeuron*/,
		unsigned /*delay*/,
		unsigned** /*targetPartition[]*/,
		unsigned** /*targetNeuron[]*/,
		float** /*weights[]*/,
		unsigned char** /*plastic[]*/)
{
	//! \todo implement this again.
	return 0;
	//return (static_cast<Network*>(network))->cm()->getRow(sourcePartition, sourceNeuron, delay,
	//		static_cast<Network*>(network)->cycle(), targetPartition, targetNeuron, weights, plastic);
}



nemo_status_t
nemo_start_simulation(NETWORK network)
{
	CATCH(network, startSimulation());
}



nemo_status_t
nemo_step(NETWORK network, unsigned fstimIdx[], size_t fstimCount)
{
	CATCH(network, stepSimulation(fstimIdx, fstimCount));
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
	CATCH_(net, *ncycles = net->readFiring(&cycles, &nidx));
	*cycles_ = const_cast<unsigned*>(&(*cycles)[0]);
	*nidx_ = const_cast<unsigned*>(&(*nidx)[0]);
	*nfired = cycles->size();
	assert(cycles->size() == nidx->size());
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


//! \todo no need to expose this in API
void
nemo_print_cycle_counters(NETWORK network)
{
	NOCATCH(network, printCycleCounters());
}



long int
nemo_elapsed_ms(NETWORK network)
{
	return NOCATCH(network, elapsed());
}


void
nemo_reset_timer(NETWORK network)
{
	// force all execution to complete first
	NOCATCH(network, syncSimulation());
	NOCATCH(network, setStart());
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
		float w_max,
		float w_min)
{
	nemo::configure_stdp(static_cast<Network*>(network)->stdpFn,
			pre_len, post_len, pre_fn, post_fn, w_max, w_min);
}


//! \todo no need to expose this in API
int
nemo_device_count()
{
	int count;
	//! \todo error handling
	cudaGetDeviceCount(&count);

	/* Even if there are no actual devices, this function will return 1, which
	 * means that device emulation can be used. We therefore need to check the
	 * major and minor device numbers as well */
	if(count == 1) {
		struct cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		if(prop.major == 9999 && prop.minor == 9999) {
			count = 0;
		}
	}
	return count;
}



void
nemo_sync_simulation(NETWORK network)
{
	NOCATCH(network, syncSimulation());
}



const char*
nemo_strerror(NETWORK network)
{
	return const_cast<char*>(static_cast<Network*>(network)->lastErrorMsg());
}

