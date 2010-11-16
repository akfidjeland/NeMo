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

#include <nemo/fixedpoint.hpp>

#include "log.cu_h"

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
#include "nvector.cu"
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
#ifdef NEMO_CUDA_KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fcm,
			fx_toFix(reward, fractionalBits),
			fx_toFix(maxWeight, fractionalBits),
			fx_toFix(minWeight, fractionalBits));
}



/*! Wrapper for the __global__ call that performs a single simulation step */
//! \todo use consistent argument ordering
__host__
void
stepSimulation(
		unsigned partitionCount,
		bool stdpEnabled,
		bool thalamicInputEnabled,
		unsigned cycle,
		uint64_t* d_recentFiring,
		float* df_neuronParameters,
		float* df_neuronState,
		unsigned* du_neuronState,
		uint32_t* d_fstim,
		fix_t* d_istim,
		uint32_t* d_fout,
		synapse_t* d_fcm,
		outgoing_addr_t* d_outgoingAddr,
		outgoing_t* d_outgoing,
		unsigned* d_incomingHeads,
		incoming_t* d_incoming,
		lq_entry_t* d_lqData,
		unsigned* d_lqFill,
		uint64_t* d_delays,
		unsigned long long* d_cc,
		size_t ccPitch)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	step<<<dimGrid, dimBlock>>>(
			stdpEnabled,
			thalamicInputEnabled,
			cycle,
			d_recentFiring,
			// neuron data
			df_neuronParameters,
			df_neuronState,
			du_neuronState,
			// spike delivery
			d_fcm,
			d_outgoingAddr, d_outgoing,
			d_incomingHeads, d_incoming,
			d_lqData, d_lqFill, d_delays,
			// stimulus
			d_fstim, // firing
			d_istim, // current
			// cycle counting
#ifdef NEMO_CUDA_KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fout);
}
