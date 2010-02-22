extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
//! \todo use RuntimeData accessors only, and get rid of these headers:
#include "ConnectivityMatrix.hpp"
#include "FiringOutput.hpp"
#include "CycleCounters.hpp"


// call function without handling exceptions
#define UNSAFE_CALL(ptr, call) static_cast<RuntimeData*>(ptr)->call


RTDATA
nemo_new_network(
		size_t maxPartitionSize,
		uint setReverse,
		uint maxReadPeriod)
{
	return new RuntimeData(maxPartitionSize, (bool) setReverse, maxReadPeriod);
}



void
nemo_delete_network(RTDATA mem)
{
	delete static_cast<RuntimeData*>(mem);
}



void
nemo_add_neuron(RTDATA rt,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	UNSAFE_CALL(rt, addNeuron(idx, a, b, c, d, u, v, sigma));
}



void
nemo_add_synapses(RTDATA rtdata,
		unsigned int source,
		unsigned int targets[],
		unsigned int delays[],
		float weights[],
		unsigned char is_plastic[],
		size_t length)
{
	UNSAFE_CALL(rtdata, addSynapses(source, targets, delays, weights, is_plastic, length));
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
	//! \todo implement this again
	return 0;
	//return (static_cast<RuntimeData*>(rtdata))->cm()->getRow(sourcePartition, sourceNeuron, delay,
	//		static_cast<RuntimeData*>(rtdata)->cycle(), targetPartition, targetNeuron, weights, plastic);
}



void
nemo_start_simulation(RTDATA rtdata)
{
	UNSAFE_CALL(rtdata, startSimulation());
}



status_t
nemo_step(RTDATA rtdata, size_t fstimCount, unsigned int fstimIdx[])
{
	return UNSAFE_CALL(rtdata, stepSimulation(fstimCount, fstimIdx));
}


void
nemo_apply_stdp(RTDATA rtdata, float reward)
{
	UNSAFE_CALL(rtdata, applyStdp(reward));
}




void
nemo_read_firing(RTDATA rtdata,
		uint** cycles,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	//! \todo expose this through RTdata
	UNSAFE_CALL(rtdata, firingOutput->readFiring(cycles, neuronIdx, nfired, ncycles));
}


void
nemo_flush_firing_buffer(RTDATA rtdata)
{
	//! \todo expose this through RTdata
	UNSAFE_CALL(rtdata, firingOutput->flushBuffer());
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


//! \todo no need to expose this in API
void
nemo_print_cycle_counters(RTDATA rtdata)
{
	UNSAFE_CALL(rtdata, printCycleCounters());
}



long int
nemo_elapsed_ms(RTDATA rtdata)
{
	return UNSAFE_CALL(rtdata, elapsed());
}


void
nemo_reset_timer(RTDATA rtdata)
{
	// force all execution to complete first
	UNSAFE_CALL(rtdata, syncSimulation());
	UNSAFE_CALL(rtdata, setStart());
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
	nemo::configure_stdp(static_cast<RuntimeData*>(rtdata)->stdpFn, pre_len, post_len, pre_fn, post_fn, w_max, w_min);
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
	UNSAFE_CALL(rtdata, syncSimulation());
}


//! \todo move the following methods into RTData, and then this file
// applyStdp
// copyToDevice
// step

