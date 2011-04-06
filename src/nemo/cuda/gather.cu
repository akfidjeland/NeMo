/*! \file gather.cu Gather kernel */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "types.h"

#include "log.cu_h"

#include "bitvector.cu"
#include "connectivityMatrix.cu"
#include "double_buffer.cu"
#include "fixedpoint.cu"
#include "globalQueue.cu"
#include "nvector.cu"


/*! \brief Add input current for a particular neuron
 *
 * The input current is stored in shared memory in a fixed-point format. This
 * necessitates overflow detection, so that we can use saturating arithmetic.
 *
 * \param[in] neuron
 *		0-based index of the target neuron
 * \param[in] current
 *		current in mA in fixed-point format
 * \param s_current
 *		shared memory vector containing current for all neurons in partition
 * \param[out] s_overflow
 *		bit vector indicating overflow status for all neurons in partition
 * \param[out] s_negative
 *		bit vector indicating the overflow sign for all neurons in partition
 *
 * \pre neuron < partition size
 * \pre all shared memory buffers have at least as many entries as partition size
 *
 * \todo add cross-reference to fixed-point format documentation
 */
__device__
void
addCurrent(nidx_t neuron,
		fix_t current,
		fix_t* s_current,
		uint32_t* s_overflow,
		uint32_t* s_negative)
{
	ASSERT(neuron < MAX_PARTITION_SIZE);
	bool overflow = fx_atomicAdd(s_current + neuron, current);
	bv_atomicSetPredicated(overflow, neuron, s_overflow);
	bv_atomicSetPredicated(overflow && fx_isNegative(current), neuron, s_negative);
#ifndef FIXPOINT_SATURATION
	ASSERT(!overflow);
#endif
}



/*! \brief Add externally provided current stimulus
 *
 * The user can provide per-neuron current stimulus
 * (\ref nemo::cuda::Simulation::addCurrentStimulus).
 *
 * \param[in] psize
 *		number of neurons in current partition
 * \param[in] pitch
 *		pitch of g_current, i.e. distance in words between each partitions data
 * \param[in] g_current
 *		global memory vector containing current for all neurons in partition.
 *		If set to NULL, no input current will be delivered.
 * \param s_current
 *		shared memory vector containing current for all neurons in partition
 * \param s_overflow
 *		bit vector indicating overflow status for all neurons in partition.
 *		Entries here may already be set and are simply OR-ed with any new entries.
 * \param s_negative
 *		bit vector indicating the overflow sign for all neurons in partition
 *		Entries here may already be set and are simply OR-ed with any new entries.
 *
 * \pre neuron < size of current partition
 * \pre all shared memory buffers have at least as many entries as the size of
 * 		the current partition
 *
 * \see nemo::cuda::Simulation::addCurrentStimulus
 */
__device__
void
addCurrentStimulus(unsigned psize,
		size_t pitch,
		const fix_t* g_current,
		fix_t* s_current,
		uint32_t* s_overflow,
		uint32_t* s_negative)
{
	if(g_current != NULL) {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			unsigned pstart = CURRENT_PARTITION * pitch;
			fix_t stimulus = g_current[pstart + neuron];
			addCurrent(neuron, stimulus, s_current, s_overflow, s_negative);
			DEBUG_MSG_SYNAPSE("c%u %u-%u: +%f (external)\n",
					s_cycle, CURRENT_PARTITION, neuron,
					fx_tofloat(g_current[pstart + neuron]));
		}
		__syncthreads();
	}
}



/*! Copy per-neuron accumulated current between two memory areas
 *
 * \param[in] current_in Per-neuron accumulated current (shared or global memory)
 * \param[out] current_out Per-neuron accumulated current (shared or global memory)
 *
 * Global memory arguments must be offset to the appropriate partition.
 */
__device__
void
copyCurrent(unsigned nNeurons, fix_t* current_in, fix_t* current_out)
{
	for(unsigned bNeuron=0; bNeuron < nNeurons; bNeuron += THREADS_PER_BLOCK) {
		unsigned neuron = bNeuron + threadIdx.x;
		current_out[neuron] = current_in[neuron];
	}
}



/*! Gather incoming current from all spikes due for delivery \e now
 *
 * The whole spike delivery process is described in more detail in \ref
 * cuda_delivery and cuda_gather.
 *
 * \param[in] cycle
 * 		Current cycle
 * \param[in] g_fcm
 *		Forward connectivity matrix in global memory
 * \param[in] g_gqFill
 *		Fill rate for global queue
 * \param[in] g_gqData
 *		Pointer to full global memory double-buffered global queue
 * \param[out] s_current
 *		per-neuron vector with accumulated current in fixed point format.
 * \param[out] s_overflow
 *		bit vector indicating overflow status for all neurons in partition.
 * \param[out] s_negative
 *		bit vector indicating the overflow sign for all neurons in partition
 */
__device__
void
gather( unsigned cycle,
		synapse_t* g_fcm,
		gq_entry_t* g_gqData,
		unsigned* g_gqFill,
		fix_t* s_current,
		uint32_t* s_overflow, // 1b per neuron overflow detection
		uint32_t* s_negative) // ditto
{
	//! \todo move init of current to here, so that we can ensure that it's zero
	/* Update incoming current in-place in fixed-point format */
	__shared__ unsigned s_incomingCount;

	//! \todo declare bit vectors here instead
	bv_clear(s_overflow);
	bv_clear(s_negative);

	if(threadIdx.x == 0) {
		//! \todo use atomicExch here instead
		size_t addr = gq_fillOffset(CURRENT_PARTITION, readBuffer(cycle));
		s_incomingCount = g_gqFill[addr];
		g_gqFill[addr] = 0;
	}
	__syncthreads();

	/* Process the incoming warps in fixed size groups */
	/*! \note Could use THREADS_PER_BLOCK here, but we're bit low on shared
	 * memory. */
#define GROUP_SIZE 128

	//! \todo could this smem be re-used?
	__shared__ synapse_t* s_warpAddress[GROUP_SIZE];

	//! \todo rename variables here
	for(unsigned groupBase = 0; groupBase < s_incomingCount; groupBase += GROUP_SIZE) {

		unsigned group = groupBase + threadIdx.x;

		/* In each loop iteration we process /up to/ GROUP_SIZE warps. For the
		 * last iteration of the outer loop we process fewer */
		__shared__ unsigned s_groupSize;
		if(threadIdx.x == 0) {
			s_groupSize =
				(groupBase + GROUP_SIZE) > s_incomingCount
				? s_incomingCount % GROUP_SIZE
				: GROUP_SIZE;
			DEBUG_MSG_SYNAPSE("c%u: group size=%u, incoming=%u\n", cycle, s_groupSize, s_incomingCount);
		}
		__syncthreads();

		if(threadIdx.x < s_groupSize) {
			gq_entry_t sgin = gq_read(readBuffer(cycle), group, g_gqData);
			s_warpAddress[threadIdx.x] = g_fcm + gq_warpOffset(sgin) * WARP_SIZE;
			DEBUG_MSG_SYNAPSE("c%u w%u -> p%u\n", cycle, gq_warpOffset(sgin), CURRENT_PARTITION);
		}

		__syncthreads();

		for(unsigned gwarp_base = 0; gwarp_base < s_groupSize; gwarp_base += WARPS_PER_BLOCK) {

			unsigned bwarp = threadIdx.x / WARP_SIZE; // warp index within a block
			unsigned gwarp = gwarp_base + bwarp;      // warp index within the global schedule

			unsigned postsynaptic;
			fix_t weight = 0;

			synapse_t* base = s_warpAddress[gwarp] + threadIdx.x % WARP_SIZE;

			/* only warps at the very end of the group are invalid here */
			if(gwarp < s_groupSize) {
				postsynaptic = targetNeuron(*base);
				weight = *((unsigned*)base + c_fcmPlaneSize);
			}

			if(weight != 0) {
				addCurrent(postsynaptic, weight, s_current, s_overflow, s_negative);
				DEBUG_MSG_SYNAPSE("c%u p?n? -> p%un%u %+f [warp %u]\n",
						s_cycle, CURRENT_PARTITION, postsynaptic,
						fx_tofloat(weight), (s_warpAddress[gwarp] - g_fcm) / WARP_SIZE);
			}
		}
		__syncthreads(); // to avoid overwriting s_groupSize
	}
}



__global__
void
gather( uint32_t cycle,
		unsigned* g_partitionSize,
		synapse_t* g_fcm,
		gq_entry_t* g_gqData,      // pitch = c_gqPitch
		unsigned* g_gqFill,
		fix_t* g_current)
{
	__shared__ fix_t s_current[MAX_PARTITION_SIZE];

	/* Per-neuron bit-vectors. See bitvector.cu for accessors */
	__shared__ uint32_t s_overflow[S_BV_PITCH];
	__shared__ uint32_t s_negative[S_BV_PITCH];

	/* Per-partition parameters */
	__shared__ unsigned s_partitionSize;

	if(threadIdx.x == 0) {
#ifdef NEMO_CUDA_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		s_current[i*THREADS_PER_BLOCK + threadIdx.x] = 0U;
	}
	__syncthreads();

	gather(cycle, g_fcm, g_gqData, g_gqFill, s_current, s_overflow, s_negative);

	/* Write back to global memory The global memory roundtrip is so that the
	 * gather and fire steps can be done in separate kernel invocations. */
	copyCurrent(s_partitionSize, s_current, g_current + CURRENT_PARTITION * c_pitch32);
}




__host__
cudaError_t
gather( cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		fix_t* d_current,
		synapse_t* d_fcm,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);
	gather<<<dimGrid, dimBlock, 0, stream>>>(cycle, d_partitionSize, d_fcm, d_gqData, d_gqFill, d_current);
	return cudaGetLastError();
}
