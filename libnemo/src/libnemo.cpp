extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
#include "except.hpp"


// call function without handling exceptions
#define NOCATCH(ptr, call) static_cast<Network*>(ptr)->call

//! \todo deal with assertion failure as well
#define CATCH(ptr, call) {                                                    \
		Network* net = static_cast<Network*>(ptr);                            \
        try {                                                                 \
			net->call;                                                        \
        } catch (DeviceAllocationException& e) {                              \
			net->setErrorMsg(e.what());                                       \
			return KERNEL_MEMORY_ERROR;                                       \
		} catch (std::exception& e) {                                         \
			net->setErrorMsg(e.what());                                       \
			return KERNEL_INVOCATION_ERROR;                                   \
        }                                                                     \
		return KERNEL_OK;                                                     \
    }



//! \todo set error locally
//! \convert error to status code
class Network : public RuntimeData {

	public :

		Network(size_t maxPartitionSize,
				bool setReverse,
				unsigned int maxReadPeriod) :
			RuntimeData(maxPartitionSize, setReverse, maxReadPeriod),
			m_errorMsg("No error") {} ;

		const char* lastErrorMsg() { return m_errorMsg.c_str(); }

		void setErrorMsg(const std::string& msg) { m_errorMsg = msg; }

	private :

		/* In addition to the runtime data, we need to keep track of the latest
		 * error and associated error message */
		std::string m_errorMsg;
};



RTDATA
nemo_new_network(
		size_t maxPartitionSize,
		uint setReverse,
		uint maxReadPeriod)
{
	return new Network(maxPartitionSize, (bool) setReverse, maxReadPeriod);
}



void
nemo_delete_network(RTDATA mem)
{
	delete static_cast<Network*>(mem);
}



status_t
nemo_add_neuron(RTDATA rt,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH(rt, addNeuron(idx, a, b, c, d, u, v, sigma));
}



status_t
nemo_add_synapses(RTDATA rtdata,
		unsigned int source,
		unsigned int targets[],
		unsigned int delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	CATCH(rtdata, addSynapses(source, targets, delays, weights, is_plastic, length));
}



size_t
nemo_get_synapses(RTDATA /*rtdata*/,
		unsigned int /*sourcePartition*/,
		unsigned int /*sourceNeuron*/,
		unsigned int /*delay*/,
		unsigned int** /*targetPartition[]*/,
		unsigned int** /*targetNeuron[]*/,
		float** /*weights[]*/,
		unsigned char** /*plastic[]*/)
{
	//! \todo implement this again.
	return 0;
	//return (static_cast<Network*>(rtdata))->cm()->getRow(sourcePartition, sourceNeuron, delay,
	//		static_cast<Network*>(rtdata)->cycle(), targetPartition, targetNeuron, weights, plastic);
}



status_t
nemo_start_simulation(RTDATA rtdata)
{
	CATCH(rtdata, startSimulation());
}



status_t
nemo_step(RTDATA rtdata, size_t fstimCount, unsigned int fstimIdx[])
{
	CATCH(rtdata, stepSimulation(fstimCount, fstimIdx));
}


status_t
nemo_apply_stdp(RTDATA rtdata, float reward)
{
	CATCH(rtdata, applyStdp(reward));
}




status_t
nemo_read_firing(RTDATA rtdata,
		uint** cycles,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	CATCH(rtdata, readFiring(cycles, neuronIdx, nfired, ncycles));
}


void
nemo_flush_firing_buffer(RTDATA rtdata)
{
	NOCATCH(rtdata, flushFiringBuffer());
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


//! \todo no need to expose this in API
void
nemo_print_cycle_counters(RTDATA rtdata)
{
	NOCATCH(rtdata, printCycleCounters());
}



long int
nemo_elapsed_ms(RTDATA rtdata)
{
	return NOCATCH(rtdata, elapsed());
}


void
nemo_reset_timer(RTDATA rtdata)
{
	// force all execution to complete first
	NOCATCH(rtdata, syncSimulation());
	NOCATCH(rtdata, setStart());
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------



void
nemo_enable_stdp(RTDATA rtdata,
		unsigned int pre_len,
		unsigned int post_len,
		float* pre_fn,
		float* post_fn,
		float w_max,
		float w_min)
{
	nemo::configure_stdp(static_cast<Network*>(rtdata)->stdpFn,
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
nemo_sync_simulation(RTDATA rtdata)
{
	NOCATCH(rtdata, syncSimulation());
}


//! \todo move the following methods into RTData, and then this file
// applyStdp
// copyToDevice
// step

