
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
#include "kernel.h"
}

#include <STDP.hpp>

#include "util.h"
#include "time.hpp"
#include "error.cu"
#include "log.hpp"
#include "L1SpikeQueue.hpp"
#include "connectivityMatrix.cu"
#include "FiringOutput.hpp"
#include "firingProbe.cu"
#include "RuntimeData.hpp"
#include "CycleCounters.hpp"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "ThalamicInput.hpp"
#include "applySTDP.cu"

#include "thalamicInput.cu"
#include "kernel.cu"
#include "stdp.cu" // only used if STDP enabled

#define STDP
#include "L1SpikeQueue.cu"
#include "step.cu"
#undef STDP
#include "L1SpikeQueue.cu"
#include "step.cu"


/* Force all asynchronously launced kernels to complete before returning */
__host__
void
syncSimulation(RTDATA rtdata)
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
}


/* Copy network data and configuration to device, if this has not already been
 * done */
void
copyToDevice(RTDATA rtdata)
{
	/* This would have been tidier if we did all the handling inside rtdata.
	 * However, problems with copying to constant memory in non-cuda code
	 * prevents this. */
	if(rtdata->deviceDirty()) {
		clearAssertions();
		rtdata->moveToDevice();
		configureKernel(rtdata);
		configurePartition(cf1_maxSynapsesPerDelay,
			rtdata->cm(CM_L1)->f_maxSynapsesPerDelay());
		//! \todo move to RSMatrix.cpp, considering that we need to call it twice (L0 and L1)
        configureReverseAddressing(
                rtdata->cm(CM_L0)->r_partitionPitch(),
                rtdata->cm(CM_L0)->r_partitionAddress(),
                rtdata->cm(CM_L0)->r_partitionStdp(),
                rtdata->cm(CM_L1)->r_partitionPitch(),
                rtdata->cm(CM_L1)->r_partitionAddress(),
                rtdata->cm(CM_L1)->r_partitionStdp());
		rtdata->setStart();
	}
}



/* Apply STDP to a single connectivity matrix */
__host__
void
applyStdp_(
		dim3 dimGrid,
		dim3 dimBlock,
		RTDATA rtdata,
		uint cmIdx,
		float reward,
		bool trace)
{
	applySTDP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			rtdata->cycleCounters->dataApplySTDP(),
			rtdata->cycleCounters->pitchApplySTDP(),
#endif
			reward,
			cmIdx,
			rtdata->stdpFn.maxWeight(),
			rtdata->stdpFn.minWeight(),
			rtdata->maxPartitionSize,
			rtdata->maxDelay(),
			//! \todo compute the address of the weight matrix here directly
			rtdata->cm(cmIdx)->df_synapses(),
			rtdata->cm(cmIdx)->df_pitch(),
			rtdata->cm(cmIdx)->df_planeSize());

	if(trace) {
		rtdata->cm(cmIdx)->df_clear(FCM_STDP_TRACE);
	}
}



__host__
void
applyStdp(RTDATA rtdata, float stdpReward)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount);

	if(rtdata->deviceDirty()) {
		return; // we haven't even started simulating yet
	}

	if(rtdata->usingStdp()) {
		if(stdpReward == 0.0f) {
			rtdata->cm(CM_L0)->clearStdpAccumulator();
			if(rtdata->haveL1Connections()) {
				rtdata->cm(CM_L1)->clearStdpAccumulator();
			}
		} else  {
			applyStdp_(dimGrid, dimBlock, rtdata, CM_L0, stdpReward, false);
			if(rtdata->haveL1Connections()) {
				applyStdp_(dimGrid, dimBlock, rtdata, CM_L1, stdpReward, false);
			}
		}
	}

	if(assertionsFailed(rtdata->partitionCount, -1)) {
		fprintf(stderr, "checking assertions\n");
		clearAssertions();
	}
}



/*! Wrapper for the __global__ call that performs a single simulation step */
__host__
status_t
step(RTDATA rtdata,
		int substeps,
		// External firing (sparse)
		size_t extFiringCount,
		const int* extFiringCIdx, 
		const int* extFiringNIdx)
{
	rtdata->step();

	copyToDevice(rtdata); // only has effect on first invocation

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount);

	static uint scycle = 0;
	DEBUG_MSG("cycle %u\n", scycle);
	scycle += 1;

	uint32_t* d_extFiring = 
		rtdata->setFiringStimulus(extFiringCount, extFiringCIdx, extFiringNIdx);

	uint32_t* d_fout = rtdata->firingOutput->step();

	if(rtdata->usingStdp()) {
		//! \todo use a function pointer here. The inputs are the same
		step_STDP<<<dimGrid, dimBlock>>>(
				substeps, 
				rtdata->cycle(),
				rtdata->recentFiring->deviceData(),
				// neuron parameters
				rtdata->neuronParameters->deviceData(),
				rtdata->thalamicInput->deviceRngState(),
				rtdata->thalamicInput->deviceSigma(),
				rtdata->neuronParameters->size(),
				// L0 forward connectivity
				rtdata->cm(CM_L0)->df_synapses(),
				rtdata->cm(CM_L0)->df_delayBits(),
				// L1 forward connectivity 
				rtdata->cm(CM_L1)->df_synapses(),
				rtdata->cm(CM_L1)->df_delayBits(),
				// L1 spike queue
				rtdata->spikeQueue->data(),
				rtdata->spikeQueue->pitch(),
				rtdata->spikeQueue->heads(),
				rtdata->spikeQueue->headPitch(),
				// firing stimulus
				d_extFiring,
				rtdata->firingStimulus->wordPitch(),
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				d_fout);
	} else {
		step_static<<<dimGrid, dimBlock>>>(
				substeps, 
				rtdata->cycle(),
				rtdata->recentFiring->deviceData(),
				rtdata->neuronParameters->deviceData(),
				rtdata->thalamicInput->deviceRngState(),
				rtdata->thalamicInput->deviceSigma(),
                //! \todo get size directly from rtdata
				rtdata->neuronParameters->size(),
				// L0 connectivity matrix
				rtdata->cm(CM_L0)->df_synapses(),
				rtdata->cm(CM_L0)->df_delayBits(),
				// L1 connectivity matrix
				rtdata->cm(CM_L1)->df_synapses(),
				rtdata->cm(CM_L1)->df_delayBits(),
				// L1 spike queue
				rtdata->spikeQueue->data(),
				rtdata->spikeQueue->pitch(),
				rtdata->spikeQueue->heads(),
				rtdata->spikeQueue->headPitch(),
				// firing stimulus
				d_extFiring,
				rtdata->firingStimulus->wordPitch(),
				// cycle counting
#ifdef KERNEL_TIMING
				rtdata->cycleCounters->data(),
				rtdata->cycleCounters->pitch(),
#endif
				// Firing output
				d_fout);
	}

    if(assertionsFailed(rtdata->partitionCount, scycle)) {
        fprintf(stderr, "checking assertions\n");
        clearAssertions();
        return KERNEL_ASSERTION_FAILURE;
    }

	cudaError_t status = cudaGetLastError();

	if(status != cudaSuccess) {
		WARNING("%s", cudaGetErrorString(status));
		LOG("", "Kernel parameters: <<<%d, %d>>>\n",
			dimGrid.x, dimBlock.x);
		return KERNEL_INVOCATION_ERROR;
	}

	return KERNEL_OK;
}
