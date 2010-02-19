#include "cycle.cu"
#include "fixedpoint.cu"
#include "bitvector.cu"

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
storeFiringOutput(uint nfired, dnidx_t* s_fired,
		uint32_t* s_dfired, uint32_t* g_dfired)
{
	bv_clear_(s_dfired);

	for(uint nbase=0; nbase < nfired; nbase += THREADS_PER_BLOCK) {
		uint i = nbase + threadIdx.x;
		uint neuron = s_fired[i];
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
fire(
	uint s_partitionSize,
	float* g_neuronParameters,
	size_t neuronParametersSize,
	// input
	float* s_current,    // input current
	// buffers
	uint32_t* s_fstim,
	uint* s_firingCount,
	dnidx_t* s_fired)    // s_NIdx, so can handle /all/ neurons firing
{
	float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	float* g_u = g_neuronParameters + STATE_U * neuronParametersSize;
	float* g_v = g_neuronParameters + STATE_V * neuronParametersSize;

	for(uint nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		uint neuron = nbase + threadIdx.x;

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
				uint i = atomicAdd(s_firingCount, 1);
				s_fired[i] = neuron;
			}

			g_v[neuron] = v;
			g_u[neuron] = u;
		}
	}
}




//=============================================================================
// Spike delivery
//=============================================================================


__device__
void
scatter(uint cycle,
		uint s_firingCount,
		dnidx_t* s_fired,
		uint* g_outgoingCount,
		outgoing_t* g_outgoing,
		uint* g_incomingHeads,
		incoming_t* g_incoming)
{
	for(uint fidxBase = 0; fidxBase < s_firingCount;
			fidxBase += THREADS_PER_BLOCK) {

		// load row lengths in parallel
		//! \note could probably use ushort here
		__shared__ uint s_len[THREADS_PER_BLOCK]; // 1KB
		uint fidx = fidxBase + threadIdx.x;
		uint presynaptic = s_fired[fidx];
		if(fidx < s_firingCount) {
			s_len[threadIdx.x] = outgoingCount(presynaptic, g_outgoingCount);
		}
		__syncthreads();

		uint fidxMax = min(fidxBase + THREADS_PER_BLOCK, s_firingCount);

		for(uint fidx = fidxBase; fidx < fidxMax; ++fidx) {

			uint presynaptic = s_fired[fidx];
			ASSERT(presynaptic < MAX_PARTITION_SIZE);

			uint len = s_len[fidx % THREADS_PER_BLOCK];

			for(uint jobBase = 0; jobBase < len; jobBase += THREADS_PER_BLOCK) {

				uint jobIdx = jobBase + threadIdx.x;

				if(jobIdx < len) {

					outgoing_t sout = outgoing(presynaptic, jobIdx, g_outgoing);

					uint delay = outgoingDelay(sout);

					ASSERT(delay > 0);

					uint targetPartition = outgoingTargetPartition(sout);
					size_t headsAddr = incomingCountAddr(targetPartition, cycle, delay);
					/*! \todo we might be able to reduce the number of atomic
					 * operations here, by writing warps going to the same
					 * target in the same go. This would be easire if we did
					 * just-in-time delivery, in which case we could do
					 * multiple smem atomics, and just a single gmem atomic */
					uint offset = atomicAdd(g_incomingHeads + headsAddr, 1);

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
gather(
		uint cycle,
		synapse_t* g_fcm,
		uint* g_incomingCount,
		incoming_t* g_incoming,
		float* s_current,
		uint32_t* s_overflow, // 1b per neuron overflow detection
		uint32_t* s_negative) // ditto
{
	//! \todo move init of current to here, so that we can ensure that it's zero
	/* Update incoming current in-place in fixed-point format */
	fix_t* s_fx_current = (fix_t*) s_current;
	__shared__ uint s_incomingCount;

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
	for(uint groupBase = 0; groupBase < s_incomingCount; groupBase += GROUP_SIZE) {

		__shared__ uint s_groupSize;

		uint group = groupBase + threadIdx.x;

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
			DEBUG_MSG("c%u w%u -> p%u\n", incomingWarpOffset(sgin), CURRENT_PARTITION);
		}

		__syncthreads();

		for(uint gwarp_base = 0; gwarp_base < s_groupSize; gwarp_base += WARPS_PER_BLOCK) {

			uint bwarp = threadIdx.x / WARP_SIZE; // warp index within a block
			uint gwarp = gwarp_base + bwarp;      // warp index within the global schedule

			uint postsynaptic;
			fix_t weight = 0;

			synapse_t* base = s_warpAddress[gwarp] + threadIdx.x % WARP_SIZE;

			/* only warps at the very end of the group are invalid here */
			if(gwarp < s_groupSize) {
				postsynaptic = targetNeuron(*base);
				weight = *((uint*)base + c_fcmPlaneSize);
			}

			if(weight != 0) {
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

	/* If any accumulators overflow, clamp to max positive or minimum value */
#ifdef FIXPOINT_SATURATION
	for(uint nbase=0; nbase < MAX_PARTITION_SIZE; nbase += THREADS_PER_BLOCK) {
		uint nidx = nbase + threadIdx.x;
		bool overflow = bv_isSet(nidx, s_overflow);
		if(overflow) {
			bool negative = bv_isSet(nidx, s_negative);
			s_fx_current[nidx] = fx_saturate(negative);
			DEBUG_MSG("c%u p%un%u input current overflow. Saturated to %+f (%08x)\n",
					s_cycle, CURRENT_PARTITION, nidx,
					fx_tofloat(s_fx_current[nidx]), s_fx_current[nidx]);
		}
	}
#endif

	/* Convert all fixed-point currents back to floating point */
	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		size_t idx = i*THREADS_PER_BLOCK + threadIdx.x;
		s_current[idx] = fx_tofloat(s_fx_current[idx]);
	}
	__syncthreads();
}
