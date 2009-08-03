
/*! \brief GPU/CUDA kernel for neural simulation using Izhikevich's model
 * 
 * The entry point for the kernel is 'step' which will do one or more
 * integrate-and-fire step.  
 *
 * \author Andreas Fidjeland
 */

#include <cutil.h>
#include <device_functions.h>
#include <stdio.h>
#include <assert.h>
extern "C" {
#include "kernel.h"
}

#include "time.hpp"
#include "error.cu"
#include "log.hpp"
#include "L1SpikeQueue.hpp"
#include "connectivityMatrix.cu"
#include "FiringProbe.hpp"
#include "firingProbe.cu"
#include "RuntimeData.hpp"
#include "CycleCounters.hpp"
#include "ConnectivityMatrix.hpp"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "ThalamicInput.hpp"
#include "applySTDP.cu"

#define SLOW_32B_INTS

#ifdef SLOW_32B_INTS
#define mul24(a,b) __mul24(a,b)
#else
#define mul24(a,b) a*b
#endif



uint
maxPartitionSize(int useSTDP)
{
    return useSTDP == 0 ? MAX_PARTITION_SIZE : MAX_PARTITION_SIZE_STDP;
}

//-----------------------------------------------------------------------------



//=============================================================================
// Double buffering
//=============================================================================

/* The current cycle indicates which half of the double buffer is for reading
 * and which is for writing */
__device__
uint
readBuffer(uint cycle)
{
    return (cycle & 0x1) ^ 0x1;
}


__device__
uint
writeBuffer(uint cycle)
{
    return cycle & 0x1;
}



//=============================================================================
// Firing 
//=============================================================================


/*! The external firing stimulus is densely packed with one bit per neuron.
 * Thus only the low-order threads need to read this data, and we need to
 * sync.  */  
__device__
void
loadExternalFiring(
        bool hasExternalInput,
		int s_partitionSize,
		size_t pitch,
		uint32_t* g_firing,
		uint32_t* s_firing)
{
	if(threadIdx.x < DIV_CEIL(s_partitionSize, 32)) {
		if(hasExternalInput) {
			s_firing[threadIdx.x] =
                g_firing[blockIdx.x * pitch + threadIdx.x];
		} else {
			s_firing[threadIdx.x] = 0;
		}
	}
	__syncthreads();
}



template<typename T>
__device__
void
loadSharedArray(int partitionSize, size_t pitch, T* g_arr, T* s_arr)
{
	for(uint nbase=0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {

		uint neuron = nbase + threadIdx.x;

		if(neuron < partitionSize) {
			s_arr[neuron] = g_arr[mul24(blockIdx.x, pitch) + neuron];
		}
	}
}



//=============================================================================
// Current buffer
//=============================================================================
// The current buffer stores (in shared memory) the accumulated current for
// each neuron in the block 
//=============================================================================


/* We need two versions of some device functions... */
#include "thalamicInput.cu"
#include "spike.cu"
#include "spikeBuffer.cu"
#include "stdp.cu" // only used if STDP enabled
#define STDP
#include "L1SpikeQueue.cu"
#include "kernel.cu"
#undef STDP
#include "L1SpikeQueue.cu"
#include "kernel.cu"


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
		configurePartition(cf0_maxSynapsesPerDelay,
			rtdata->cm(CM_L0)->f_maxSynapsesPerDelay());
		configurePartition(cr0_maxSynapsesPerNeuron,
			rtdata->cm(CM_L0)->r_maxPartitionPitch());
		configurePartition(cf1_maxSynapsesPerDelay,
			rtdata->cm(CM_L1)->f_maxSynapsesPerDelay());
		configurePartition(cr1_maxSynapsesPerNeuron,
			rtdata->cm(CM_L1)->r_maxPartitionPitch());
		rtdata->configureSTDP();
        rtdata->setStart();
	}
}




/*! Clear STDP accumulators for a connectivity matrix */
__host__
void
clearSTDPAccumulator(dim3 dimGrid, dim3 dimBlock, RTDATA rtdata, uint cmIdx)
{
	rtdata->cm(cmIdx)->dr_clear(RCM_STDP);
}




__host__
void
applySTDP(
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
			rtdata->stdpMaxWeight(),
			rtdata->maxPartitionSize,
			rtdata->maxDelay(),
			rtdata->cm(cmIdx)->df_synapses(),
			rtdata->cm(cmIdx)->df_pitch(),
			rtdata->cm(cmIdx)->df_planeSize(),
			rtdata->cm(cmIdx)->dr_synapses(),
			rtdata->cm(cmIdx)->dr_pitch(),
			rtdata->cm(cmIdx)->dr_planeSize());

	if(trace) {
		rtdata->cm(cmIdx)->df_clear(FCM_STDP_TRACE);
	}
}


/*! Wrapper for the __global__ call that performs a single simulation step */
__host__
status_t
step(	ushort cycle,
        //! \todo make all these unsigned
		int substeps,
		int doApplySTDP,
		float stdpReward,
		// External firing (sparse)
		size_t extFiringCount,
		const int* extFiringCIdx, 
		const int* extFiringNIdx, 
		// Run-time data
		RTDATA rtdata)
{
	rtdata->firingProbe->checkOverflow();
	rtdata->step();

	copyToDevice(rtdata); // only has effect on first invocation

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount);

#ifdef VERBOSE
	static uint scycle = 0;
	fprintf(stdout, "cycle %u\n", scycle++);
#endif

	if(rtdata->usingSTDP() && doApplySTDP) {
		if(stdpReward == 0.0f) {
			clearSTDPAccumulator(dimGrid, dimBlock, rtdata, CM_L0);
			if(rtdata->haveL1Connections()) {
				clearSTDPAccumulator(dimGrid, dimBlock, rtdata, CM_L1);
			}
		} else  {
			applySTDP(dimGrid, dimBlock, rtdata, CM_L0, stdpReward, false);
			if(rtdata->haveL1Connections()) {
				applySTDP(dimGrid, dimBlock, rtdata, CM_L1, stdpReward, false);
			}
		}
	}

	uint32_t* d_extFiring = 
		rtdata->setFiringStimulus(extFiringCount, extFiringCIdx, extFiringNIdx);

	if(rtdata->usingSTDP()) {
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
				// L0 reverse connectivity
				rtdata->cm(CM_L0)->dr_synapses(),
				// L1 reverse connectivity
				rtdata->cm(CM_L1)->dr_synapses(),
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
				cycle,
				rtdata->firingProbe->deviceBuffer(),
				rtdata->firingProbe->deviceNextFree());
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
				cycle,
				rtdata->firingProbe->deviceBuffer(),
				rtdata->firingProbe->deviceNextFree());
	}

    if(assertionsFailed(rtdata->partitionCount, cycle)) {
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
