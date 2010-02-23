extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
#include "DeviceAssertions.hpp"
#include "except.hpp"



//! \todo hard-code spaces
/* We cannot propagate exceptions via the C API, so convert to an error code
 * instead */
#define CATCH(ptr, call) {                                                    \
        Network* net = static_cast<Network*>(ptr);                            \
        try {                                                                 \
            net->call;                                                        \
        } catch (DeviceAllocationException& e) {                              \
            net->setErrorMsg(e.what());                                       \
            return KERNEL_MEMORY_ERROR;                                       \
        } catch (KernelInvocationError& e) {                                  \
            net->setErrorMsg(e.what());                                       \
            return KERNEL_INVOCATION_ERROR;                                   \
        } catch (DeviceAssertionFailure& e) {                                 \
            net->setErrorMsg(e.what());                                       \
            return KERNEL_ASSERTION_FAILURE;                                  \
        } catch (std::exception& e) {                                         \
            net->setErrorMsg(e.what());                                       \
            return KERNEL_UNKNOWN_ERROR;                                      \
        }                                                                     \
        return KERNEL_OK;                                                     \
    }


//! \todo enforce no throw in the class interface
/* Call function without handling exceptions */
#define NOCATCH(ptr, call) static_cast<Network*>(ptr)->call



class Network : public RuntimeData {

	public :

		Network(bool setReverse, unsigned maxReadPeriod) :
			RuntimeData(setReverse, maxReadPeriod),
			m_errorMsg("No error") { } ;

		Network(bool setReverse,
				unsigned maxReadPeriod,
				unsigned maxPartitionSize) :
			RuntimeData(setReverse, maxReadPeriod, maxPartitionSize),
			m_errorMsg("No error") { } ;

		const char* lastErrorMsg() { return m_errorMsg.c_str(); }

		void setErrorMsg(const std::string& msg) { m_errorMsg = msg; }

	private :

		/* In addition to the runtime data, we need to keep track of the latest
		 * error and associated error message */
		std::string m_errorMsg;
};


RTDATA
nemo_new_network(unsigned setReverse, unsigned maxReadPeriod)
{
	return new Network((bool) setReverse, maxReadPeriod);
}



RTDATA
nemo_new_network_(
		unsigned setReverse,
		unsigned maxReadPeriod,
		unsigned maxPartitionSize)
{
	return new Network((bool) setReverse, maxReadPeriod, maxPartitionSize);
}



void
nemo_delete_network(RTDATA mem)
{
	delete static_cast<Network*>(mem);
}



status_t
nemo_add_neuron(RTDATA rt,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH(rt, addNeuron(idx, a, b, c, d, u, v, sigma));
}



status_t
nemo_add_synapses(RTDATA rtdata,
		unsigned source,
		unsigned targets[],
		unsigned delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	CATCH(rtdata, addSynapses(source, targets, delays, weights, is_plastic, length));
}



size_t
nemo_get_synapses(RTDATA /*rtdata*/,
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
	//return (static_cast<Network*>(rtdata))->cm()->getRow(sourcePartition, sourceNeuron, delay,
	//		static_cast<Network*>(rtdata)->cycle(), targetPartition, targetNeuron, weights, plastic);
}



status_t
nemo_start_simulation(RTDATA rtdata)
{
	CATCH(rtdata, startSimulation());
}



status_t
nemo_step(RTDATA rtdata, size_t fstimCount, unsigned fstimIdx[])
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
		unsigned pre_len,
		unsigned post_len,
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
