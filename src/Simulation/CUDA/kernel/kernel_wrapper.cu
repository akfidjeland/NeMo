
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
#include "L1SpikeQueue.cu"
#include "FiringProbe.hpp"
#include "firingProbe.cu"
#include "RuntimeData.hpp"
#include "CycleCounters.hpp"
#include "ConnectivityMatrix.hpp"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "ThalamicInput.hpp"

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


/*! \param chunk 
 *		Each thread processes n neurons. The chunk is the index (<n) of the
 *		neuron currently processed by the thread.
 * \return 
 *		true if the thread should be active when processing the specified
 * 		neuron, false otherwise.
 */
__device__
bool 
activeNeuron(int chunk, int s_partitionSize)
{
	return threadIdx.x + chunk*THREADS_PER_BLOCK < s_partitionSize;
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



//=============================================================================
// Current buffer
//=============================================================================
// The current buffer stores (in shared memory) the accumulated current for
// each neuron in the block 
//=============================================================================


/* We need two versions of some device functions... */
#include "thalamicInput.cu"
#define STDP
#include "kernel.cu"
#undef STDP
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
__host__
void
configureDevice(RTDATA rtdata)
{
	/* This would have been tidier if we did all the handling inside rtdata.
	 * However, problems with copying to constant memory in non-cuda code
	 * prevents this. */
	if(rtdata->deviceDirty()) {
        clearAssertions();
		rtdata->moveToDevice();
		configureKernel(rtdata);
		configurePartition(c_maxL0SynapsesPerDelay, 
			rtdata->cm(CM_L0)->f_maxSynapsesPerDelay());
		configurePartition(c_maxL0RevSynapsesPerDelay, 
			rtdata->cm(CM_L0)->r_maxSynapsesPerDelay());
		configurePartition(c_maxL1SynapsesPerDelay,
			rtdata->cm(CM_L1)->f_maxSynapsesPerDelay());
		configurePartition(c_maxL1RevSynapsesPerDelay,
			rtdata->cm(CM_L1)->r_maxSynapsesPerDelay());
		if(rtdata->usingSTDP()) {
			configureStdp(
				rtdata->m_stdpTauP,
				rtdata->m_stdpTauD,
				rtdata->m_stdpAlphaP,
				rtdata->m_stdpAlphaD);
		}
        rtdata->setStart();
	}
}




/*! Clear STDP accumulators for a connectivity matrix */
__host__
void
clearSTDPAccumulator(dim3 dimGrid, dim3 dimBlock, RTDATA rtdata, uint cmIdx)
{
	rtdata->cm(cmIdx)->df_clear(FCM_STDP_LTD);
	rtdata->cm(cmIdx)->dr_clear(RCM_STDP_LTP);
}




__host__
void
applySTDP(dim3 dimGrid,
		dim3 dimBlock,
		RTDATA rtdata,
		uint cmIdx,
		float reward,
		bool trace)
{
	reorderLTP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			rtdata->cycleCounters->dataReorderSTDP(),
			rtdata->cycleCounters->pitchReorderSTDP(),
#endif
			rtdata->maxPartitionSize,
			rtdata->maxDelay(),
			rtdata->pitch32(),
			rtdata->cm(cmIdx)->dr_delayBits(),
			rtdata->cm(cmIdx)->df_synapses(),
			rtdata->cm(cmIdx)->df_pitch(),
			rtdata->cm(cmIdx)->df_planeSize(),
			rtdata->cm(cmIdx)->dr_synapses(),
			rtdata->cm(cmIdx)->dr_pitch(),
			rtdata->cm(cmIdx)->dr_planeSize());

	if(trace) {
		rtdata->cm(cmIdx)->df_clear(FCM_STDP_TRACE);
	}

	applySTDP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			rtdata->cycleCounters->dataApplySTDP(),
			rtdata->cycleCounters->pitchApplySTDP(),
#endif
			reward,
			rtdata->stdpMaxWeight(),
			rtdata->maxPartitionSize,
			rtdata->maxDelay(),
			rtdata->pitch32(),
			rtdata->cm(cmIdx)->df_delayBits(),
			rtdata->cm(cmIdx)->df_synapses(),
			rtdata->cm(cmIdx)->df_pitch(),
			rtdata->cm(cmIdx)->df_planeSize(),
			trace);

	rtdata->cm(cmIdx)->df_clear(FCM_STDP_LTP);
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

	configureDevice(rtdata); // only done on first invocation

	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(rtdata->partitionCount);

#ifdef VERBOSE
	static uint scycle = 0;
	fprintf(stdout, "cycle %u/%u\n", scycle++, rtdata->stdpCycle());
#endif

	if(rtdata->usingSTDP() && doApplySTDP) {
		if(stdpReward == 0.0f) {
			clearSTDPAccumulator(dimGrid, dimBlock, rtdata, CM_L0);
		} else  {
			applySTDP(dimGrid, dimBlock, rtdata, CM_L0, stdpReward, false);
		}
	}

	uint32_t* d_extFiring = 
		rtdata->setFiringStimulus(extFiringCount, extFiringCIdx, extFiringNIdx);

	if(rtdata->usingSTDP()) {
		step_STDP<<<dimGrid, dimBlock>>>(
				substeps, 
				rtdata->cycle(),
				rtdata->recentFiring->deviceData(),
				// STDP
				rtdata->recentArrivals->deviceData(),
				rtdata->stdpCycle(),
				rtdata->cm(CM_L0)->dr_synapses(),
				rtdata->cm(CM_L0)->dr_delayBits(),
				rtdata->cm(CM_L1)->dr_delayBits(),
				// neuron parameters
				rtdata->neuronParameters->deviceData(),
				rtdata->thalamicInput->deviceRngState(),
				rtdata->thalamicInput->deviceSigma(),
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
