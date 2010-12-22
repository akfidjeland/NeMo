
#include "log.cu_h"
#include "fixedpoint.cu"
#include "bitvector.cu"
#include "localQueue.cu"


//=============================================================================
// Firing
//=============================================================================



/*! Set per-neuron bit-vector for fired neurons in both shared and global memory
 *
 * \param[in] nfired
 *		Number of neurons in current partition which fired this cycle.
 * \param[in] s_fired
 *		Vector of indices of the fired neuron. The first \a nfired entries
 *		should be set.
 * \param[out] s_dfired
 *		Per-neuron bit-vector in shared memory for fired neurons.
 * \param[out] g_dfired
 *		Per-neuron bit-vector in global memory for fired neurons.
 */
__device__
void
storeDenseFiringOutput(unsigned nfired, nidx_dt* s_fired,
		uint32_t* s_dfired, uint32_t* g_dfired)
{
	bv_clear_(s_dfired);

	for(unsigned nbase=0; nbase < nfired; nbase += THREADS_PER_BLOCK) {
		unsigned i = nbase + threadIdx.x;
		unsigned neuron = s_fired[i];
		bv_atomicSetPredicated(i < nfired, neuron, s_dfired);
	}
	__syncthreads();

	bv_copy(s_dfired, g_dfired + CURRENT_PARTITION * c_bv_pitch);
}



/*! Store sparse firing in global memory buffer
 *
 * The global memory roundtrip is required to support having 'fire' and
 * 'scatter' in separate kernels.
 *
 * \param[in] nFired number of neurons in this partition which fired this cycle
 * \param[in] s_fired shared memory vector of the relevant neuron indices.
 * \param[out] g_nFired global memory per-partition vector of firing counts
 * \param[out] g_fired global memory per-neuron vector of fired neuron indices.
 * 		For each partition, only the first \a nFired entries contain valid data.
 */
__device__
void
storeSparseFiring(unsigned nFired, nidx_dt* s_fired, unsigned* g_nFired, nidx_dt* g_fired)
{
	for(unsigned b=0; b < nFired; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < nFired) {
			g_fired[CURRENT_PARTITION * c_pitch32 + i] = s_fired[i];
		}
	}

	if(threadIdx.x == 0) {
		g_nFired[CURRENT_PARTITION] = nFired;
	}
}



/*! Load sparse firing from global memory buffer
 *
 * The global memory roundtrip is required to support having 'fire' and
 * 'scatter' in separate kernels.
 *
 * \param[in] g_nFired global memory per-partition vector of firing counts
 * \param[in] g_fired global memory per-neuron vector of fired neuron indices.
 * \param[out] s_nFired number of neurons in this partition which fired this cycle
 * \param[out] s_fired shared memory vector of the relevant neuron indices.
 * 		Only the first \a nFired entries contain valid data.
 */
__device__
void
loadSparseFiring(unsigned* g_nFired, nidx_dt* g_fired, unsigned* s_nFired, nidx_dt* s_fired)
{
	__shared__ unsigned nFired;
	if(threadIdx.x == 0) {
		nFired = g_nFired[CURRENT_PARTITION];
	}
	__syncthreads();

	for(unsigned b=0; b < nFired; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < nFired) {
			s_fired[i] = g_fired[CURRENT_PARTITION * c_pitch32 + i];
		}
	}
	__syncthreads();
}



/*! The external firing stimulus is (possibly) provided in a per-neuron
 * bit-vector */
__device__
void
loadFiringInput(uint32_t* g_firing, uint32_t* s_firing)
{
	if(g_firing != NULL) {
		bv_copy(g_firing + CURRENT_PARTITION * c_bv_pitch, s_firing);
	} else {
		bv_clear(s_firing);
	}
	__syncthreads();
}



/*! Update state of all neurons
 *
 * Update the state of all neurons in partition according to the equations in
 * Izhikevich's 2003 paper based on
 *
 * - the neuron parameters (a-d)
 * - the neuron state (u, v)
 * - input current (from other neurons, random input current, or externally provided)
 * - per-neuron specific firing stimulus
 *
 * The neuron state is updated using the Euler method.
 *
 * \param[in] s_partitionSize
 *		number of neurons in current partition
 * \param[in] g_neuronParameters
 *		global memory containing neuron parameters (see \ref nemo::cuda::Neurons)
 * \param[in] g_neuronState
 *		global memory containing neuron state (see \ref nemo::cuda::Neurons)
 * \param[in] s_current
 *		shared memory vector containing input current for all neurons in
 *		partition
 * \param[in] s_fstim
 *		shared memory bit vector where set bits indicate neurons which should
 *		be forced to fire
 * \param[out] s_nFired
 *		output variable which will be set to the number of	neurons which fired
 *		this cycle
 * \param[out] s_fired
 *		shared memory vector containing local indices of neurons which fired.
 *		s_fired[0:s_nFired-1] will contain valid data, whereas remaining
 *		entries may contain garbage.
 */
__device__
void
fire(
	unsigned s_partitionSize,
	float* g_neuronParameters,
	float* g_neuronState,
	// input
	float* g_current,    // input current
	// buffers
	uint32_t* s_fstim,
	// output
	unsigned* s_nFired,
	nidx_dt* s_fired)    // s_NIdx, so can handle /all/ neurons firing
{
	size_t neuronParametersSize = PARTITION_COUNT * c_pitch32;
	float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	float* g_u = g_neuronState + STATE_U * neuronParametersSize;
	float* g_v = g_neuronState + STATE_V * neuronParametersSize;

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		if(neuron < s_partitionSize) {

			float v = g_v[neuron];
			float u = g_u[neuron];
			float a = g_a[neuron];
			float b = g_b[neuron];
			float I = g_current[neuron];

			/* n sub-steps for numerical stability, with u held */
			bool fired = false;
			for(int j=0; j < 4; ++j) {
				if(!fired) { 
					v += 0.25f * ((0.04f*v + 5.0f) * v + 140.0f - u + I);
					/*! \todo: could pre-multiply this with a, when initialising memory */
					u += 0.25f * (a * ( b*v - u ));
					fired = v >= 30.0f;
				} 
			}

			bool forceFiring = bv_isSet(neuron, s_fstim); // (smem broadcast)

			if(fired || forceFiring) {

				/* Only a subset of the neurons fire and thus require c/d
				 * fetched from global memory. One could therefore deal with
				 * all the fired neurons separately. This was found, however,
				 * to slow down the fire step by 50%, due to extra required
				 * synchronisation.  */
				//! \todo could probably hard-code c
				v = g_c[neuron];
				u += g_d[neuron];

				DEBUG_MSG_NEURON("c%u %u-%u fired (forced: %u)\n",
						s_cycle, CURRENT_PARTITION, neuron, forceFiring);

				//! \todo consider *only* updating this here, and setting u and v separately
				unsigned i = atomicAdd(s_nFired, 1);

				/* can overwrite current as long as i < neuron. See notes below
				 * on synchronisation and declaration of s_current/s_fired. */
				s_fired[i] = neuron;
			}

			g_v[neuron] = v;
			g_u[neuron] = u;
		}

		/* synchronise to ensure accesses to s_fired and s_current (which use
		 * the same underlying buffer) do not overlap. Even in the worst case
		 * (all neurons firing) the write to s_fired will be at least one
		 * before the first unconsumed s_current entry. */
		__syncthreads();
	}
}




//=============================================================================
// Spike delivery
//=============================================================================



/*! Enque all recently fired neurons in the local queue
 *
 * See the section on \ref cuda_local_delivery "local spike delivery" for more
 * details.
 *
 * \param cycle
 * \param nFired number of valid entries in s_fired
 * \param s_fired shared memory buffer containing the recently fired neurons
 * \param g_delays delay bits for each neuron
 * \param g_fill queue fill for local queue
 * \param g_queue global memory for the local queue
 */
__device__
void
scatterLocal(
		unsigned cycle,
		unsigned nFired,
		const nidx_dt* s_fired,
		uint64_t* g_delays,
		unsigned* g_fill,
		lq_entry_t* g_queue)
{
	/* This shared memory vector is quite small, so no need to reuse */
	__shared__ unsigned s_fill[MAX_DELAY];

	lq_loadQueueFill(g_fill, s_fill);
	__syncthreads();

	/*! \todo do more than one neuron at a time. We can deal with
	 * THREADS_PER_BLOCK/MAX_DELAY per iteration. */
	for(unsigned iFired = 0; iFired < nFired; iFired++) {

		unsigned neuron = s_fired[iFired];

		__shared__ uint64_t delayBits;
		/*! \todo could load more delay data in one go */
		if(threadIdx.x == 0) {
			delayBits = nv_load64(neuron, 0, g_delays);
		}
		__syncthreads();

		//! \todo handle MAX_DELAY > THREADS_PER_BLOCK
		unsigned delay0 = threadIdx.x;
		if(delay0 < MAX_DELAY) {
			bool delaySet = (delayBits >> uint64_t(delay0)) & 0x1;
			if(delaySet) {
				/* This write operation will almost certainly be non-coalesced.
				 * It would be possible to stage data in smem, e.g. one warp
				 * per queue slot. 64 slots would require 64 x 32 x 4B = 8kB.
				 * Managaging this data can be costly, however, as we need to
				 * flush buffers as we go. */
				lq_enque(neuron, cycle, delay0, s_fill, g_queue);
				DEBUG_MSG_SYNAPSE("c%u[local scatter]: enque n%u d%u\n", cycle, neuron, delay0+1);
			}
		}
	}
	__syncthreads();
	lq_storeQueueFill(s_fill, g_fill);
}



/*! Echange spikes between partitions
 *
 * See the section on \ref cuda_global_delivery "global spike delivery" for
 * more details.
 */
__device__
void
scatterGlobal(unsigned cycle,
		unsigned* g_lqFill,
		lq_entry_t* g_lq,
		outgoing_addr_t* g_outgoingAddr,
		outgoing_t* g_outgoing,
		unsigned* g_gqFill,
		gq_entry_t* g_gqData)
{
	__shared__ unsigned s_fill[MAX_PARTITION_COUNT]; // 512

	/* Instead of iterating over fired neurons, load all fired data from a
	 * single local queue entry. Iterate over the neuron/delay pairs stored
	 * there. */
	__shared__ unsigned s_nLq;

	if(threadIdx.x == 0) {
		s_nLq = lq_getAndClearCurrentFill(cycle, g_lqFill);
	}
	__syncthreads();

	for(unsigned bLq = 0; bLq < s_nLq; bLq += THREADS_PER_BLOCK) {

		unsigned iLq = bLq + threadIdx.x;

		//! \todo share this memory with other stages
#ifdef NEMO_CUDA_DEBUG_TRACE
		__shared__ lq_entry_t s_lq[THREADS_PER_BLOCK];   // 1KB
#endif
		__shared__ unsigned s_offset[THREADS_PER_BLOCK]; // 1KB
		__shared__ unsigned s_len[THREADS_PER_BLOCK];    // 1KB

		s_len[threadIdx.x] = 0;

		/* Load local queue entries (neuron/delay pairs) and the associated
		 * outgoing lengths into shared memory */
		if(iLq < s_nLq) {
			ASSERT(iLq < c_lqPitch);
			lq_entry_t entry = g_lq[lq_offset(cycle, 0) + iLq];
#ifdef NEMO_CUDA_DEBUG_TRACE
			s_lq[threadIdx.x] = entry;
#endif
			short delay0 = entry.y;
			ASSERT(delay0 < MAX_DELAY);

			short neuron = entry.x;
			ASSERT(neuron < MAX_PARTITION_SIZE);

			/* Outgoing counts is cachable. It is not too large and is runtime
			 * constant. It is too large for constant memory however. The
			 * alternatives are thus texture memory or the L1 cache (on Fermi) */
			outgoing_addr_t addr = outgoingAddr(neuron, delay0, g_outgoingAddr);
			s_offset[threadIdx.x] = addr.x;
			s_len[threadIdx.x] = addr.y;
			ASSERT(s_len[threadIdx.x] <= c_outgoingPitch);
			DEBUG_MSG_SYNAPSE("c%u[global scatter]: dequeued n%u d%u from local queue (%u warps from %u)\n",
					cycle, neuron, delay0, s_len[threadIdx.x], s_offset[threadIdx.x]);
		}
		__syncthreads();

		/* Now loop over all the entries we just loaded from the local queue.
		 * Read a number of entries in one go, if possible. Note that a large
		 * spread in the range of outgoing row lengths (e.g. one extremely long
		 * one) will adveresely affect performance here. */
		unsigned jLqMax = min(THREADS_PER_BLOCK, s_nLq-bLq);
		for(unsigned jbLq = 0; jbLq < jLqMax; jbLq += c_outgoingStep) {

			/* jLq should be in [0, 256) so that we can point to s_len
			 * e.g.     0,8,16,24,...,248 + 0,1,...,8 */
			unsigned jLq = jbLq + threadIdx.x / c_outgoingPitch;
			ASSERT(jLq < THREADS_PER_BLOCK);

			/* There may be more than THREADS_PER_BLOCK entries in this
			 * outgoing row, although the common case should be just a single
			 * loop iteration here */
			unsigned nOut = s_len[jLq];
			if(threadIdx.x < PARTITION_COUNT) {
				s_fill[threadIdx.x] = 0;
			}
			__syncthreads();

			/* Load row of outgoing data (specific to neuron/delay pair) */
			unsigned iOut = threadIdx.x % c_outgoingPitch;
			unsigned targetPartition = 0;
			unsigned warpOffset = 0;
			unsigned localOffset = 0;
			bool valid = bLq + jLq < s_nLq && iOut < nOut;
			if(valid) {
				outgoing_t sout = g_outgoing[s_offset[jLq] + iOut];
				targetPartition = outgoingTargetPartition(sout);
				ASSERT(targetPartition < PARTITION_COUNT);
				warpOffset = outgoingWarpOffset(sout);
				ASSERT(warpOffset != 0);
				localOffset = atomicAdd(s_fill + targetPartition, 1);
			}
			__syncthreads();

			/* Update s_fill to store actual offset */
			if(threadIdx.x < PARTITION_COUNT) {
				size_t fillAddr = gq_fillOffset(threadIdx.x, writeBuffer(cycle));
				s_fill[threadIdx.x] = atomicAdd(g_gqFill + fillAddr, s_fill[threadIdx.x]);
			}
			__syncthreads();

			if(valid) {
				unsigned offset = s_fill[targetPartition] + localOffset;
				ASSERT(offset < c_incomingPitch);
				size_t base = gq_bufferStart(targetPartition, writeBuffer(cycle));
				g_gqData[base + offset] = warpOffset;
				DEBUG_MSG_SYNAPSE("c%u[global scatter]: enqueued warp %u (p%un%u -> p%u with d%u) to global queue (buffer entry %u/%lu)\n",
						cycle, warpOffset,
						CURRENT_PARTITION, s_lq[jLq].x, targetPartition, s_lq[jLq].y,
						offset, c_incomingPitch);
				/* The writes to the global queue are non-coalesced. It would
				 * be possible to stage this data in smem for each partition.
				 * However, this would require a fair amount of smem (1), and
				 * handling buffer overflow is complex and introduces
				 * sequentiality. Overall, it's probably not worth it.
				 *
				 * (1) 128 partitions, warp-sized buffers, 4B/entry = 16KB
				 */
			}
			__syncthreads(); // to protect s_fill
		}
		__syncthreads(); // to protect s_len
	}
}





//=============================================================================
// Step simulation
//=============================================================================


/*! \brief Perform a single simulation step
 *
 * A simulation step consists of five main parts:
 *
 * - gather incoming current from presynaptic firing for each neuron (\ref gather)
 * - add externally or internally provided input current for each neuron
 * - update the neuron state (\ref fire)
 * - enque outgoing spikes for neurons which fired (\ref scatterLocal and \ref scatterGlobal)
 * - accumulate STDP statistics
 *
 * The data structures involved in each of these stages are documentated more
 * with the individual functions and in \ref cuda_delivery.
 */

__global__
void
fireAndScatter (
		bool stdpEnabled,
		uint32_t cycle,
		uint64_t* g_recentFiring,
		// neuron state
		float* gf_neuronParameters,
		float* gf_neuronState,
		// spike delivery
		outgoing_addr_t* g_outgoingAddr,
		outgoing_t* g_outgoing,
		gq_entry_t* g_gqData,      // pitch = c_gqPitch
		unsigned* g_gqFill,
		lq_entry_t* g_lqData,      // pitch = c_lqPitch
		unsigned* g_lqFill,
		uint64_t* g_delays,        // pitch = c_pitch64
		// firing stimulus
		uint32_t* g_fstim,
		float* g_current,
#ifdef NEMO_CUDA_KERNEL_TIMING
		// cycle counting
		cycle_counter_t* g_cycleCounters,
		//! \todo move to cmem
		size_t ccPitch,
#endif
		uint32_t* g_firingOutput, // dense output, already offset to current cycle
		unsigned* g_nFired,       // device-only buffer
		nidx_dt* g_fired)         // device-only buffer, sparse output
{
	SET_COUNTER(s_ccMain, 0);

	__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];
	__shared__ uint32_t s_N1A[S_BV_PITCH];

	__shared__ unsigned s_nFired;
	__shared__ unsigned s_partitionSize;

	if(threadIdx.x == 0) {
#ifdef NEMO_CUDA_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_nFired = 0;
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	uint32_t* s_fstim = s_N1A;
	loadFiringInput(g_fstim, s_fstim);

	fire( s_partitionSize,
			gf_neuronParameters + CURRENT_PARTITION * c_pitch32,
			gf_neuronState + CURRENT_PARTITION * c_pitch32,
			g_current + CURRENT_PARTITION * c_pitch32,
			s_fstim,
			&s_nFired,
			s_fired);

	__syncthreads();

	uint32_t* s_dfired = s_N1A;
	storeDenseFiringOutput(s_nFired, s_fired, s_dfired, g_firingOutput);
	storeSparseFiring(s_nFired, s_fired, g_nFired, g_fired);

	// add kernel boundary here

	loadSparseFiring(g_nFired, g_fired, &s_nFired, s_fired);

	scatterLocal(cycle, s_nFired, s_fired, g_delays, g_lqFill, g_lqData);

	scatterGlobal(cycle,
			g_lqFill,
			g_lqData,
			g_outgoingAddr,
			g_outgoing,
			g_gqFill,
			g_gqData);

	if(stdpEnabled) {
		loadStdpParameters_();
		updateSTDP_(
				cycle,
				s_dfired,
				g_recentFiring,
				c_pitch64,
				s_partitionSize,
				cr_address, cr_stdp, cr_pitch,
				s_fired);
	}
}
