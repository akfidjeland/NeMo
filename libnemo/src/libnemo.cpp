extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
//! \todo use RuntimeData accessors only, and get rid of these headers:
#include "ConnectivityMatrix.hpp"
#include "FiringOutput.hpp"
#include "CycleCounters.hpp"



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
	delete mem;
}




void
nemo_add_neuron(RTDATA rt,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	rt->addNeuron(idx, a, b, c, d, u, v, sigma);
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
	rtdata->cm()->setRow(
		source,
		targets,
		delays,
		weights,
		is_plastic,
		length);
}




size_t
nemo_get_synapses(RTDATA rtdata,
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetPartition[],
		unsigned int* targetNeuron[],
		float* weights[],
		unsigned char* plastic[])
{
	return rtdata->cm()->getRow(sourcePartition, sourceNeuron, delay,
			rtdata->cycle(), targetPartition, targetNeuron, weights, plastic);
}


//-----------------------------------------------------------------------------
// Generated firing
//-----------------------------------------------------------------------------

void
nemo_read_firing(RTDATA rtdata,
		uint** cycles,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	rtdata->firingOutput->readFiring(cycles, neuronIdx, nfired, ncycles);
}


void
nemo_flush_firing_buffer(RTDATA rtdata)
{
	rtdata->firingOutput->flushBuffer();
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


//! \todo no need to expose this in API
void
nemo_print_cycle_counters(RTDATA rtdata)
{
	rtdata->cycleCounters->printCounters();
}



long int
nemo_elapsed_ms(RTDATA rtdata)
{
	return rtdata->elapsed();
}


void
nemo_reset_timer(RTDATA rtdata)
{
	// force all execution to complete first
	rtdata->syncSimulation();
	rtdata->setStart();
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
	nemo::configure_stdp(rtdata->stdpFn, pre_len, post_len, pre_fn, post_fn, w_max, w_min);
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
	rtdata->syncSimulation();
}


//! \todo move the following methods into RTData, and then this file
// applyStdp
// copyToDevice
// step

