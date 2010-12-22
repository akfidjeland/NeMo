//! \file kernel.cu

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

#include "kernel.cu_h"
#include "log.cu_h"


/*! Per-partition size
 *
 * Different partitions need not have exactly the same size. The exact size of
 * each partition is stored in constant memory, so that per-neuron loops can do
 * the correct minimum number of iterations
 */
__constant__ unsigned c_partitionSize[MAX_PARTITION_COUNT];


#include "device_assert.cu"
#include "bitvector.cu"
#include "double_buffer.cu"
#include "connectivityMatrix.cu"
#include "cycleCounting.cu"
#include "outgoing.cu"
#include "globalQueue.cu"
#include "nvector.cu"
#include "stdp.cu"
#include "step.cu"

#include "applySTDP.cu"
#include "gather.cu"


/*! Set partition size for each partition in constant memory
 * \see c_partitionSize */
__host__
cudaError
configurePartitionSize(const unsigned* d_partitionSize, size_t len)
{
	//! \todo set padding to zero
	assert(len <= MAX_PARTITION_COUNT);
	return cudaMemcpyToSymbol(
			c_partitionSize,
			(void*) d_partitionSize,
			MAX_PARTITION_COUNT*sizeof(unsigned),
			0, cudaMemcpyHostToDevice);
}



__host__
void
applyStdp(
		cycle_counter_t* d_cc,
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
cudaError_t
stepSimulation(
		unsigned partitionCount,
		bool stdpEnabled,
		unsigned cycle,
		uint64_t* d_recentFiring,
		float* df_neuronParameters,
		float* df_neuronState,
		uint32_t* d_fstim,
		float* d_current,
		uint32_t* d_fout,
		outgoing_addr_t* d_outgoingAddr,
		outgoing_t* d_outgoing,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill,
		lq_entry_t* d_lqData,
		unsigned* d_lqFill,
		uint64_t* d_delays,
		cycle_counter_t* d_cc,
		size_t ccPitch)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	fireAndScatter<<<dimGrid, dimBlock>>>(
			stdpEnabled,
			cycle,
			d_recentFiring,
			// neuron data
			df_neuronParameters,
			df_neuronState,
			// spike delivery
			d_outgoingAddr, d_outgoing,
			d_gqData, d_gqFill,
			d_lqData, d_lqFill, d_delays,
			// stimulus
			d_fstim, // firing stimulus
			d_current, // internal input current
			// cycle counting
#ifdef NEMO_CUDA_KERNEL_TIMING
			d_cc, ccPitch,
#endif
			d_fout);

	return cudaGetLastError();
}
