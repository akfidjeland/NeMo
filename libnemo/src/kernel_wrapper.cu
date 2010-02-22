
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

#include <STDP.hpp>

#include "util.h"
#include "time.hpp"
#include "device_assert.cu"
#include "log.hpp"
#include "connectivityMatrix.cu"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "applySTDP.cu"
#include "outgoing.cu"
#include "incoming.cu"

#include "thalamicInput.cu"
#include "kernel.cu"
#include "stdp.cu"
#include "step.cu"



__host__
void
applyStdp(
		unsigned long long* d_cc,
		size_t ccPitch,
		uint partitionCount,
		uint fractionalBits,
		synapse_t* d_fcm,
		const nemo::STDP<float>& stdpFn,
		float reward)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	applySTDP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fcm,
			fixedPoint(reward, fractionalBits),
			fixedPoint(stdpFn.maxWeight(), fractionalBits),
			fixedPoint(stdpFn.minWeight(), fractionalBits));

	if(assertionsFailed(partitionCount, 0)) {
		clearAssertions();
	}
}



/*! Wrapper for the __global__ call that performs a single simulation step */
//! \todo use consitent argument ordering
__host__
void
stepSimulation(
		uint partitionCount,
		bool usingStdp,
		uint cycle,
		uint64_t* d_recentFiring,
		float* d_neuronState,
		unsigned* d_rngState,
		float* d_rngSigma,
		uint32_t* d_fstim,
		uint32_t* d_fout,
		synapse_t* d_fcm,
		uint* d_outgoingCount,
		outgoing_t* d_outgoing,
		uint* d_incomingHeads,
		incoming_t* d_incoming,
		unsigned long long* d_cc,
		size_t ccPitch)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	step<<<dimGrid, dimBlock>>>(
			usingStdp,
			cycle,
			d_recentFiring,
			// neuron parameters
			d_neuronState,
			d_rngState, d_rngSigma,
			// spike delivery
			d_fcm,
			d_outgoingCount, d_outgoing,
			d_incomingHeads, d_incoming,
			// firing stimulus
			d_fstim,
			// cycle counting
#ifdef KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fout);
}
