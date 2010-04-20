/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <assert.h>

#include "log.cu_h"
#include "util.h"

#include "device_assert.cu"
#include "bitvector.cu"
#include "double_buffer.cu"
#include "connectivityMatrix.cu"
#include "partitionConfiguration.cu"
#include "cycleCounting.cu"
#include "applySTDP.cu"
#include "outgoing.cu"
#include "incoming.cu"
#include "thalamicInput.cu"
#include "stdp.cu"
#include "step.cu"



__host__
void
applyStdp(
		unsigned long long* d_cc,
		size_t ccPitch,
		unsigned partitionCount,
		unsigned fractionalBits,
		synapse_t* d_fcm,
		float maxWeight,
		float minWeight,
		float reward)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	applySTDP_<<<dimGrid, dimBlock>>>(
#ifdef KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fcm,
			fx_toFix(reward, fractionalBits),
			fx_toFix(maxWeight, fractionalBits),
			fx_toFix(minWeight, fractionalBits));
}



/*! Wrapper for the __global__ call that performs a single simulation step */
//! \todo use consitent argument ordering
__host__
void
stepSimulation(
		unsigned partitionCount,
		bool usingStdp,
		unsigned cycle,
		uint64_t* d_recentFiring,
		float* d_neuronState,
		unsigned* d_rngState,
		float* d_rngSigma,
		uint32_t* d_fstim,
		uint32_t* d_fout,
		synapse_t* d_fcm,
		unsigned* d_outgoingCount,
		outgoing_t* d_outgoing,
		unsigned* d_incomingHeads,
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
