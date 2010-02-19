extern "C" {
#include "libnemo.h"
}

#include "RuntimeData.hpp"
//! \todo use RuntimeData accessors only, and get rid of these headers:
#include "ConnectivityMatrix.hpp"
#include "FiringOutput.hpp"
#include "CycleCounters.hpp"



RTDATA
allocRuntimeData(
		size_t maxPartitionSize,
		uint setReverse,
		uint maxReadPeriod)
{
	return new RuntimeData(maxPartitionSize, (bool) setReverse, maxReadPeriod);
}


void
freeRuntimeData(RTDATA mem)
{
	delete mem;
}




void
addNeuron(RTDATA rt,
		unsigned int idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	rt->addNeuron(idx, a, b, c, d, u, v, sigma);
}



void
addSynapses(RTDATA rtdata,
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



void
setCMDRow(RTDATA rtdata,
		unsigned int sourceNeuron,
		unsigned int delay,
		unsigned int* targetNeuron,
		float* weights,
		unsigned char* isPlastic,
		size_t length)
{
	std::vector<unsigned int> delays(length, delay);
	rtdata->cm()->setRow(
		sourceNeuron,
		targetNeuron,
		&delays[0],
		weights,
		isPlastic,
		length);
}



size_t
getCMDRow(RTDATA rtdata,
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
readFiring(RTDATA rtdata,
		uint** cycles,
		uint** neuronIdx,
		uint* nfired,
		uint* ncycles)
{
	rtdata->firingOutput->readFiring(cycles, neuronIdx, nfired, ncycles);
}


void
flushFiringBuffer(RTDATA rtdata)
{
	rtdata->firingOutput->flushBuffer();
}


size_t
allocatedDeviceMemory(RTDATA rt)
{
	return rt->d_allocated();
}





//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


void
printCycleCounters(RTDATA rtdata)
{
	rtdata->cycleCounters->printCounters();
}



long int
elapsedMs(RTDATA rtdata)
{
	return rtdata->elapsed();
}


void
resetTimer(RTDATA rtdata)
{
	// force all execution to complete first
	rtdata->syncSimulation();
	rtdata->setStart();
}



//-----------------------------------------------------------------------------
// STDP
//-----------------------------------------------------------------------------



void
enableStdp(RTDATA rtdata,
		unsigned int pre_len,
		unsigned int post_len,
		float* pre_fn,
		float* post_fn,
		float w_max,
		float w_min)
{
	nemo::configure_stdp(rtdata->stdpFn, pre_len, post_len, pre_fn, post_fn, w_max, w_min);
}


int
deviceCount()
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
syncSimulation(RTDATA rtdata)
{
	rtdata->syncSimulation();
}


//! \todo move the following methods into this file
// applyStdp
// copyToDevice
// step

