
#include "log.cu_h"
#include "fixedpoint.cu"
#include "bitvector.cu"
#include "localQueue.cu"


//=============================================================================
// Firing
//=============================================================================



/*! Set per-neuron bit-vector for fired neurons in both shared and global memory
 *
 * \param nfired
 *		Number of neurons in current partition which fired this cycle.
 * \param s_fired
 *		Vector of indices of the fired neuron. The first \a nfired entries
 *		should be set.
 * \param s_dfired
 *		Per-neuron bit-vector in shared memory for fired neurons.
 * \param g_dfired
 *		Per-neuron bit-vector in global memory for fired neurons.
 */
__device__
void
storeFiringOutput(unsigned nfired, nidx_dt* s_fired,
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


/* Add current to current vector for a particular neuron and update fixed-point
 * overflow indicators */
__device__
void
addCurrent(nidx_t neuron,
		fix_t current,
		fix_t* s_current,
		uint32_t* s_overflow,
		uint32_t* s_negative)
{
	bool overflow = fx_atomicAdd(s_current + neuron, current);
	bv_atomicSetPredicated(overflow, neuron, s_overflow);
	bv_atomicSetPredicated(overflow && fx_isNegative(current), neuron, s_negative);
#ifndef FIXPOINT_SATURATION
	ASSERT(!overflow);
#endif
}



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



__device__
void
fire(
	unsigned s_partitionSize,
	float* g_neuronParameters,
	float* g_neuronState,
	// input
	float* s_current,    // input current
	// buffers
	uint32_t* s_fstim,
	// output
	unsigned* s_firingCount,
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
			float I = s_current[neuron];

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
				unsigned i = atomicAdd(s_firingCount, 1);

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
 * \param nFired number of valid entries in s_fired
 * \param s_fired shared memory buffer containing the recently fired neurons
 * \param g_delays delay bits for each neuron
 * \param g_queue global memory for the local queue
 * \param s_counts shared memory buffer for the per delay-slot queue fill. This
 *        should contain at leaset MAX_DELAY elements.
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
				DEBUG_MSG_SYNAPSE("c%u lq: delay bits %0lx\n", cycle, delayBits);
				DEBUG_MSG_SYNAPSE("c%u lq: add n%u d%u\n", cycle, neuron, delay0+1);
			}
		}
	}
	__syncthreads();
	lq_storeQueueFill(s_fill, g_fill);
}



__device__
void
scatterGlobal(unsigned cycle,
		unsigned* g_lqFill,
		lq_entry_t* g_lq,
		unsigned* g_outgoingCount,
		outgoing_t* g_outgoing,
		unsigned* g_incomingHeads,
		incoming_t* g_incoming)
{
	/* Instead of iterating over fired neurons, load all fired data from a
	 * single local queue entry. Iterate over the neuron/delay pairs stored
	 * there. */
	__shared__ unsigned nLq;
	if(threadIdx.x == 0) {
		nLq = lq_getAndClearCurrentFill(cycle, g_lqFill);
	}
	__syncthreads();
	DEBUG_THREAD_MSG(0, "c%u local queue fill = %u\n", cycle, nLq);

	for(unsigned bLq = 0; bLq < nLq; bLq += THREADS_PER_BLOCK) {

		unsigned iLq = bLq + threadIdx.x;

		//! \todo share this memory with other stages
		__shared__ lq_entry_t s_lq[THREADS_PER_BLOCK]; // 1KB
		__shared__ unsigned s_len[THREADS_PER_BLOCK];  // 1KB

		s_len[threadIdx.x] = 0;

		/* Load local queue entries (neuron/delay pairs) and the associated
		 * outgoing lengths into shared memory */
		if(iLq < nLq) {
			ASSERT(iLq < c_lqPitch);
			lq_entry_t entry = g_lq[lq_offset(cycle, 0) + iLq];
			/*! \todo store precomputed address here instead of neuron/delay pair */
			s_lq[threadIdx.x] = entry;
			/* For now this returns just the length. The outgoing data should
			 * later be compacted, in which case this data should contain an
			 * offset/length pair instead */ 
			short delay0 = entry.y;
			/*! \todo we should be able to cache the outgoing counts */
			s_len[threadIdx.x] = outgoingCount(entry.x, delay0, g_outgoingCount);
			DEBUG_MSG_SYNAPSE("c%u[global scatter]: dequeued n%u d%u from local queue (%u warps)\n",
					cycle, entry.x, delay0, s_len[threadIdx.x]);
		}
		__syncthreads();

		/* Now loop over all the entries we just loaded from the local queue */ 
		unsigned jLqMax = min(THREADS_PER_BLOCK, nLq-bLq);
		for(unsigned jLq = 0; jLq < jLqMax; ++jLq) {

			/* There may be more than THREADS_PER_BLOCK entries in this
			 * outgoing row, although the common case should be just a single
			 * loop iteration here */
			unsigned nOut = s_len[jLq];
			for(unsigned bOut = 0; bOut < nOut; bOut += THREADS_PER_BLOCK) {

				/* Load row of outgoing data (specific to neuron/delay pair) */
				unsigned iOut = bOut + threadIdx.x;
				if(iOut < nOut) {
					short sourceNeuron = s_lq[jLq].x;
					short delay0 = s_lq[jLq].y;
					outgoing_t sout = outgoing(sourceNeuron, delay0, iOut, g_outgoing); // coalesced
					unsigned targetPartition = outgoingTargetPartition(sout);
					/*! \todo use consistent naming here: fill rather than count/heads */
					size_t headsAddr = incomingCountAddr(targetPartition, writeBuffer(cycle));
					unsigned offset = atomicAdd(g_incomingHeads + headsAddr, 1);
					ASSERT(offset < c_incomingPitch);
					size_t base = incomingBufferStart(targetPartition, writeBuffer(cycle));
					g_incoming[base + offset] = make_incoming(outgoingWarpOffset(sout));

					DEBUG_MSG_SYNAPSE("c%u[global scatter]: enqueued warp %u (p%un%u -> p%u with d%u) to global queue (buffer entry %u/%lu)\n",
							cycle, outgoingWarpOffset(sout),
							CURRENT_PARTITION, sourceNeuron, targetPartition, s_lq[jLq].y,
							offset, c_incomingPitch);
					/* These memory operations are non-coalesced. However, we
					 * might be able to stage data in smem for each partition.
					 * This would require 128 buffers x 32 x 4B = 16kB, so  the
					 * whole smem on 10-series. We might be better of with a 2k
					 * partition size. */
					/* If we're careless with the outgoing data we could
					 * overflow smem buffers for outgoing entries */ 
				}
				// flush any buffers which are full
			}
		}
		__syncthreads(); // to protect s_len
	}
}



__device__
void
gather( unsigned cycle,
		synapse_t* g_fcm,
		unsigned* g_incomingCount,
		incoming_t* g_incoming,
		float* s_current,
		uint32_t* s_overflow, // 1b per neuron overflow detection
		uint32_t* s_negative) // ditto
{
	//! \todo move init of current to here, so that we can ensure that it's zero
	/* Update incoming current in-place in fixed-point format */
	fix_t* s_fx_current = (fix_t*) s_current;
	__shared__ unsigned s_incomingCount;

	bv_clear(s_overflow);
	bv_clear(s_negative);

	if(threadIdx.x == 0) {
		//! \todo use atomicExch here instead
		size_t addr = incomingCountAddr(CURRENT_PARTITION, readBuffer(cycle));
		s_incomingCount = g_incomingCount[addr];
		g_incomingCount[addr] = 0;
	}
	__syncthreads();

	/*! \note Could use THREADS_PER_BLOCK here, but we're bit low on shared
	 * memory. */
#define GROUP_SIZE 128

	//! \todo could this smem be re-used?
	__shared__ synapse_t* s_warpAddress[GROUP_SIZE];

	//! \todo rename variables here
	for(unsigned groupBase = 0; groupBase < s_incomingCount; groupBase += GROUP_SIZE) {

		__shared__ unsigned s_groupSize;

		unsigned group = groupBase + threadIdx.x;

		if(threadIdx.x == 0) {
			s_groupSize =
				(group + GROUP_SIZE) > s_incomingCount
				? s_incomingCount % GROUP_SIZE
				: GROUP_SIZE;
			DEBUG_MSG_SYNAPSE("c%u: group size=%u, incoming=%u\n", cycle, s_groupSize, s_incomingCount);
		}
		__syncthreads();

		if(threadIdx.x < s_groupSize) {
			incoming_t sgin = getIncoming(readBuffer(cycle), group, g_incoming);
			s_warpAddress[threadIdx.x] = g_fcm + incomingWarpOffset(sgin) * WARP_SIZE;
			DEBUG_MSG_SYNAPSE("c%u w%u -> p%u\n", cycle, incomingWarpOffset(sgin), CURRENT_PARTITION);
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
				addCurrent(postsynaptic, weight, s_fx_current, s_overflow, s_negative);
				DEBUG_MSG_SYNAPSE("c%u p?n? -> p%un%u %+f [warp %u]\n",
						s_cycle, CURRENT_PARTITION, postsynaptic,
						fx_tofloat(weight), (s_warpAddress[gwarp] - g_fcm) / WARP_SIZE);
			}
		}
		__syncthreads(); // to avoid overwriting s_groupSize
	}
}





//=============================================================================
// Step simulation
//=============================================================================


/*! Combined integrate and fire using sparse connectivity matrix, a single step
 * updates the state (u and v) of each neuron and produces spikes to be used in
 * the next simulation cycle.
 *
 * The number of neurons per block provided to the kernel is always warp-
 * aligned. This means that some threads do useless work, but at no cost. Using
 * a warp-aligned neuron number simplifies the control when the number of
 * neurons is not an exact multiple of the number of threads per block.
 *
 * The parameters (a, b, c, and d) can be set for each individual neuron and
 * should be pre-loaded in global memory before this kernel is invoked.
 */
__global__
void
step (
		bool stdpEnabled,
		bool thalamicInputEnabled,
		uint32_t cycle,
		uint64_t* g_recentFiring,
		// neuron state
		float* gf_neuronParameters,
		float* gf_neuronState,
		unsigned* gu_neuronState,
		// spike delivery
		synapse_t* g_fcm,
		unsigned* g_outgoingCount,
		outgoing_t* g_outgoing,
		unsigned* g_incomingHeads,
		incoming_t* g_incoming,
		lq_entry_t* g_lqData,      // pitch = c_lqPitch
		unsigned* g_lqFill,
		uint64_t* g_delays,        // pitch = c_pitch64
		// firing stimulus
		uint32_t* g_fstim,
		fix_t* g_istim,
#ifdef NEMO_CUDA_KERNEL_TIMING
		// cycle counting
		//! \todo change to uint64_t
		unsigned long long* g_cycleCounters,
		//! \todo move to cmem
		size_t ccPitch,
#endif
		uint32_t* firingOutput) // already offset to current cycle
{
	SET_COUNTER(s_ccMain, 0);

	/* Per-neuron buffers */

	/* We're re-using the same bit of shared memory here for both the current
	 * (per-neuron) and the list of firing (per *fired*-neuron). These are both
	 * used in the firing step (with different addressing). We end up writing
	 * firing data to the lower part of this array while the upper region still
	 * contains unconsumed current. This is safe since the read part s_fired is
	 * offset from the start, and we use a synchronisation in the fire step.
	 * See notes there as well.
	 *
	 * While this might seem borderline insane, it frees up
	 * 4*MAX_PARTITION_SIZE bytes of shared memory, which is a big deal */
	__shared__ uint32_t s_N32[MAX_PARTITION_SIZE + THREADS_PER_BLOCK];
	float* s_current = (float*) s_N32 + THREADS_PER_BLOCK;
	nidx_dt* s_fired = (nidx_dt*) s_N32;

	// in practice the above works out the same as
	//__shared__ float s_current[MAX_PARTITION_SIZE];
	//__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];

	/* Per-neuron bit-vectors. See bitvector.cu for accessors */
	__shared__ uint32_t s_N1A[S_BV_PITCH];
	__shared__ uint32_t s_N1B[S_BV_PITCH];

	/* Per-partition parameters */
	__shared__ unsigned s_partitionSize;
	__shared__ unsigned s_firingCount;

	if(threadIdx.x == 0) {
#ifdef NEMO_CUDA_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_firingCount = 0;
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		s_current[i*THREADS_PER_BLOCK + threadIdx.x] = 0.0f;
	}
	SET_COUNTER(s_ccMain, 1);

	uint32_t* s_overflow = s_N1A;
	uint32_t* s_negative = s_N1B;

	gather(cycle, g_fcm, g_incomingHeads, g_incoming, s_current, s_overflow, s_negative);

	SET_COUNTER(s_ccMain, 2);

	addCurrentStimulus(s_partitionSize, c_pitch32, g_istim, (fix_t*) s_current, s_overflow, s_negative);
	fx_arrSaturatedToFloat(s_overflow, s_negative, (fix_t*) s_current, s_current);

	SET_COUNTER(s_ccMain, 3);

	/* Generating random input current really ought to be done /before/
	 * providing the input current (for better performance in MPI backend).
	 * However, we need to either provide fixed-point random input or do an
	 * additional conversion inside the thalamic input code in order for this
	 * to work. */
	if(thalamicInputEnabled) {
		thalamicInput(s_partitionSize, c_pitch32,
				gu_neuronState, gf_neuronParameters, s_current);
	}

	SET_COUNTER(s_ccMain, 4);

	uint32_t* s_fstim = s_N1A;
	loadFiringInput(g_fstim, s_fstim);

	fire( s_partitionSize,
			gf_neuronParameters + CURRENT_PARTITION * c_pitch32,
			gf_neuronState + CURRENT_PARTITION * c_pitch32,
			s_current,
			s_fstim,
			&s_firingCount,
			s_fired);

	__syncthreads();

	uint32_t* s_dfired = s_N1A;
	storeFiringOutput(s_firingCount, s_fired, s_dfired, firingOutput);

	SET_COUNTER(s_ccMain, 5);

	scatterLocal(cycle, s_firingCount, s_fired, g_delays, g_lqFill, g_lqData);

	SET_COUNTER(s_ccMain, 6);

	scatterGlobal(cycle,
			g_lqFill,
			g_lqData,
			g_outgoingCount,
			g_outgoing,
			g_incomingHeads,
			g_incoming);

	SET_COUNTER(s_ccMain, 7);

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

	SET_COUNTER(s_ccMain, 8);

	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
