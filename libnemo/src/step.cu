/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "cycle.cu"
#include "fixedpoint.cu"
#include "bitvector.cu"


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



__device__
void
addCurrentStimulus(unsigned psize, size_t pitch, const float* g_current, float* s_current)
{
	if(g_current != NULL) {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			size_t pstart = CURRENT_PARTITION * pitch;
			s_current[neuron] += g_current[pstart + neuron];
		}
		__syncthreads();
	}
}



__device__
void
fire(
	unsigned s_partitionSize,
	float* g_neuronParameters,
	// input
	float* s_current,    // input current
	// buffers
	uint32_t* s_fstim,
	// output
	unsigned* s_firingCount,
	nidx_dt* s_fired)    // s_NIdx, so can handle /all/ neurons firing
{
	//! \todo put s_pitch32 in cmem
	size_t neuronParametersSize = PARTITION_COUNT * s_pitch32;
	float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	float* g_u = g_neuronParameters + STATE_U * neuronParametersSize;
	float* g_v = g_neuronParameters + STATE_V * neuronParametersSize;

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

				DEBUG_MSG("c%u %u-%u fired (forced: %u) (thread %u)\n",
						s_cycle, CURRENT_PARTITION, neuron,
						forceFiring, threadIdx.x);

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


__device__
void
scatter(unsigned cycle,
		unsigned s_firingCount,
		nidx_dt* s_fired,
		unsigned* g_outgoingCount,
		outgoing_t* g_outgoing,
		unsigned* g_incomingHeads,
		incoming_t* g_incoming)
{
	for(unsigned fidxBase = 0; fidxBase < s_firingCount;
			fidxBase += THREADS_PER_BLOCK) {

		// load row lengths in parallel
		//! \note could probably use ushort here
		__shared__ unsigned s_len[THREADS_PER_BLOCK]; // 1KB
		unsigned fidx = fidxBase + threadIdx.x;
		unsigned presynaptic = s_fired[fidx];
		if(fidx < s_firingCount) {
			s_len[threadIdx.x] = outgoingCount(presynaptic, g_outgoingCount);
		}
		__syncthreads();

		unsigned fidxMax = min(fidxBase + THREADS_PER_BLOCK, s_firingCount);

		for(unsigned fidx = fidxBase; fidx < fidxMax; ++fidx) {

			unsigned presynaptic = s_fired[fidx];
			ASSERT(presynaptic < MAX_PARTITION_SIZE);

			unsigned len = s_len[fidx % THREADS_PER_BLOCK];

			for(unsigned jobBase = 0; jobBase < len; jobBase += THREADS_PER_BLOCK) {

				unsigned jobIdx = jobBase + threadIdx.x;

				if(jobIdx < len) {

					outgoing_t sout = outgoing(presynaptic, jobIdx, g_outgoing);

					unsigned delay = outgoingDelay(sout);

					ASSERT(delay > 0);

					unsigned targetPartition = outgoingTargetPartition(sout);
					size_t headsAddr = incomingCountAddr(targetPartition, cycle, delay);
					/*! \todo we might be able to reduce the number of atomic
					 * operations here, by writing warps going to the same
					 * target in the same go. This would be easier if we did
					 * just-in-time delivery, in which case we could do
					 * multiple smem atomics, and just a single gmem atomic */
					unsigned offset = atomicAdd(g_incomingHeads + headsAddr, 1);

					ASSERT(offset < c_incomingPitch);

					size_t base = incomingBufferStart(targetPartition, cycle, delay);
					g_incoming[base + offset] = make_incoming(outgoingWarpOffset(sout));

					DEBUG_MSG("c%u spike warp p%un%u -> p%u (delay %u) (buffer entry %u/%u)\n",
							cycle, CURRENT_PARTITION, presynaptic, targetPartition, delay,
							offset, c_incomingPitch);
				}
			}
		}
		__syncthreads(); // so s_len is not updated
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
		size_t addr = incomingCountAddr(CURRENT_PARTITION, cycle, 0);
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
			DEBUG_MSG("c%u: group size=%u, incoming=%u\n", cycle, s_groupSize, s_incomingCount);
		}
		__syncthreads();

		if(threadIdx.x < s_groupSize) {
			incoming_t sgin = getIncoming(cycle, group, g_incoming);
			s_warpAddress[threadIdx.x] = g_fcm + incomingWarpOffset(sgin) * WARP_SIZE;
			DEBUG_MSG("c%u w%u -> p%u\n", cycle, incomingWarpOffset(sgin), CURRENT_PARTITION);
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
				//! \todo combine this into a single operation
				bool overflow = fx_atomicAdd(s_fx_current + postsynaptic, weight);
				bv_atomicSetPredicated(overflow, postsynaptic, s_overflow);
				bv_atomicSetPredicated(overflow && fx_isNegative(weight),
						postsynaptic, s_negative);
#ifndef FIXPOINT_SATURATION
				ASSERT(!overflow);
#endif
				DEBUG_MSG("c%u p?n? -> p%un%u %+f (%08x)\n",
						s_cycle, CURRENT_PARTITION, postsynaptic,
						fx_tofloat(weight), weight);
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
		uint32_t cycle,
		uint64_t* g_recentFiring,
		// neuron state
		float* g_neuronParameters,
		unsigned* g_rngState,
		float* g_rngSigma,			//! \todo combine with g_neuronParameters
		// spike delivery
		synapse_t* g_fcm,
		unsigned* g_outgoingCount,
		outgoing_t* g_outgoing,
		unsigned* g_incomingHeads,
		incoming_t* g_incoming,
		// firing stimulus
		uint32_t* g_fstim,
		float* g_istim,
#ifdef KERNEL_TIMING
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
	__shared__ uint32_t s_N1A[MAX_PARTITION_SIZE/32];
	__shared__ uint32_t s_N1B[MAX_PARTITION_SIZE/32];

	/* Per-partition parameters */
	__shared__ unsigned s_partitionSize;
	__shared__ unsigned s_firingCount;

	if(threadIdx.x == 0) {
#ifdef __DEVICE_EMULATION__
		s_cycle = cycle;
#endif
		s_firingCount = 0;
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	loadNetworkParameters();

	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		s_current[i*THREADS_PER_BLOCK + threadIdx.x] = 0.0f;
	}
	SET_COUNTER(s_ccMain, 1);

	uint32_t* s_overflow = s_N1A;
	uint32_t* s_negative = s_N1B;

	gather(cycle, g_fcm, g_incomingHeads, g_incoming, s_current, s_overflow, s_negative);
	fx_arrSaturatedToFloat(s_overflow, s_negative, (fix_t*) s_current, s_current);

	SET_COUNTER(s_ccMain, 2);

	uint32_t* s_fstim = s_N1A;
	loadFiringInput(g_fstim, s_fstim);
	addCurrentStimulus(s_partitionSize, s_pitch32, g_istim, s_current);

	SET_COUNTER(s_ccMain, 3);

	/* Generating random input current really ought to be done /before/
	 * providing the input current (for better performance in MPI backend).
	 * However, we need to either provide fixed-point random input or do an
	 * additional conversion inside the thalamic input code in order for this
	 * to work. */
	if(g_rngState != NULL && g_rngSigma != NULL) {
		thalamicInput(s_partitionSize, s_pitch32, g_rngState, g_rngSigma, s_current);
	}

	SET_COUNTER(s_ccMain, 4);

	fire( s_partitionSize,
			g_neuronParameters + CURRENT_PARTITION * s_pitch32,
			s_current,
			s_fstim,
			&s_firingCount,
			s_fired);

	__syncthreads();

	uint32_t* s_dfired = s_N1A;
	storeFiringOutput(s_firingCount, s_fired, s_dfired, firingOutput);

	SET_COUNTER(s_ccMain, 5);

	scatter(
			cycle,
			s_firingCount,
			s_fired,
			g_outgoingCount,
			g_outgoing,
			g_incomingHeads,
			g_incoming);

	SET_COUNTER(s_ccMain, 6);

	if(stdpEnabled) {
		loadStdpParameters_();
		updateSTDP_(
				cycle,
				s_dfired,
				g_recentFiring,
				s_pitch64,
				s_partitionSize,
				cr_address, cr_stdp, cr_pitch,
				s_fired);
	}

	SET_COUNTER(s_ccMain, 7);

	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
