
/*! \brief GPU/CUDA kernel for neural simulation using Izhikevich's model
 * 
 * The entry point for the kernel is 'step' which will do one or more
 * integrate-and-fire step.  
 *
 * \author Andreas Fidjeland
 */

#include <device_functions.h>
#include <stdio.h>
#include <assert.h>
extern "C" {
#include "libnemo.h"
}

#include <STDP.hpp>

#include "util.h"
#include "time.hpp"
#include "error.cu"
#include "log.hpp"
#include "connectivityMatrix.cu"
#include "FiringOutput.hpp"
#include "RuntimeData.hpp"
#include "CycleCounters.hpp"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "ThalamicInput.hpp"
#include "applySTDP.cu"
#include "outgoing.cu"
#include "incoming.cu"

#include "thalamicInput.cu"
#include "kernel.cu"
#include "stdp.cu" // only used if STDP enabled

#define STDP
#include "step.cu"
#undef STDP
#include "step.cu"



__host__
void
applyStdp(RuntimeData* rtdata, float reward)
{
	uint fb = rtdata->cm()->fractionalBits();
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount());

	applySTDP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			rtdata->cycleCounters->dataApplySTDP(),
			rtdata->cycleCounters->pitchApplySTDP(),
#endif
			rtdata->cm()->d_fcm(),
			fixedPoint(reward, fb),
			fixedPoint(rtdata->stdpFn.maxWeight(), fb),
			fixedPoint(rtdata->stdpFn.minWeight(), fb));

	if(assertionsFailed(rtdata->partitionCount(), -1)) {
		clearAssertions();
	}
}



/*! Wrapper for the __global__ call that performs a single simulation step */
//! \todo don't return status_t here. Only deal with this in API layer
__host__
status_t
stepSimulation(RuntimeData* rtdata, uint32_t* d_fstim, uint32_t* d_fout)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount());

	//! \todo use cycle number from rtdata insteda
	static uint scycle = 0;
	DEBUG_MSG("cycle %u\n", scycle);
	scycle += 1;

	if(rtdata->usingStdp()) {
		//! \todo use a function pointer here. The inputs are the same
		step_STDP<<<dimGrid, dimBlock>>>(
				rtdata->cycle(),
				rtdata->recentFiring->deviceData(),
				// neuron parameters
				rtdata->d_neurons(),
				rtdata->thalamicInput->deviceRngState(),
				rtdata->thalamicInput->deviceSigma(),
				rtdata->neuronVectorLength(),
				// spike delivery
				rtdata->cm()->d_fcm(),
				rtdata->cm()->outgoingCount(),
				rtdata->cm()->outgoing(),
				rtdata->cm()->incomingHeads(),
				rtdata->cm()->incoming(),
				// firing stimulus
				d_fstim,
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				d_fout);
	} else {
		step_static<<<dimGrid, dimBlock>>>(
				rtdata->cycle(),
				rtdata->recentFiring->deviceData(),
				rtdata->d_neurons(),
				rtdata->thalamicInput->deviceRngState(),
				rtdata->thalamicInput->deviceSigma(),
				rtdata->neuronVectorLength(),
				// spike delivery
				rtdata->cm()->d_fcm(),
				rtdata->cm()->outgoingCount(),
				rtdata->cm()->outgoing(),
				rtdata->cm()->incomingHeads(),
				rtdata->cm()->incoming(),
				// firing stimulus
				d_fstim,
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				// Firing output
				d_fout);
	}

    if(assertionsFailed(rtdata->partitionCount(), scycle)) {
        fprintf(stderr, "checking assertions\n");
        clearAssertions();
        return KERNEL_ASSERTION_FAILURE;
    }

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess) {
		WARNING("c%u %s", rtdata->cycle(), cudaGetErrorString(status));
		LOG("", "Kernel parameters: <<<%d, %d>>>\n",
			dimGrid.x, dimBlock.x);
		return KERNEL_INVOCATION_ERROR;
	}

	return KERNEL_OK;
}
