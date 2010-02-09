#include "cycle.cu"

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



template<typename T>
__device__
void
loadSharedArray(int partitionSize, size_t pitch, T* g_arr, T* s_arr)
{
	for(uint nbase=0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {
		uint neuron = nbase + threadIdx.x;
		if(neuron < partitionSize) {
			s_arr[neuron] = g_arr[(blockIdx.x * pitch) + neuron];
		}
	}
}



//=============================================================================
// Shared memory buffers
//=============================================================================



__device__
void
updateHistory(uint s_partitionSize,
		uint32_t* s_dfired,
		uint64_t* s_recentFiring,
		uint64_t* g_recentFiring)
{
	for(uint nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {
		uint neuron = nbase + threadIdx.x;
		if(neuron < s_partitionSize) {
			g_recentFiring[neuron] =
				(s_recentFiring[neuron] << 1) | (didFire(neuron, s_dfired) ? 0x1 : 0x0);
		}
	}
	__syncthreads();
}




__device__
void
fire(
	uint s_partitionSize,
	uint substeps,
	float substepMult, // substepMul * substeps = 1
	size_t pitch1, //! \todo move into shared memory along with other pitches
	float* g_neuronParameters,
	size_t neuronParametersSize,
	// input
	float* s_current,    // input current
	// buffers
	uint32_t* s_fstim,   // s_T16, so larger than needed
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
			for(int j=0; j<substeps; ++j) {
				if(!fired) { 
					v += substepMult * ((0.04f*v + 5.0f) * v + 140.0f - u + I);
					/*! \todo: could pre-multiply this with a, when initialising memory */
					u += substepMult * (a * ( b*v - u ));
					fired = v >= 30.0f;
				} 
			}

			/* s_fstim accessed using broadcast */
			bool forceFiring = (s_fstim[neuron/32] >> (neuron % 32)) & 0x1;

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
					g_incoming[base + offset] =
						make_incoming(
								outgoingWarpOffset(sout),
								outgoingTargetBits(sout));

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
		float* s_current)
{
	__shared__ uint s_incomingCount;

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

	// could use 3b of warpAddress
	__shared__ uchar s_warpCommit[GROUP_SIZE]; // in range 0-8
	__shared__ uint32_t s_targetBits[GROUP_SIZE];

#define SCHEDULING_THREADS (GROUP_SIZE / WARPS_PER_BLOCK)
	__shared__ uchar s_warpCommitCount[SCHEDULING_THREADS];

	//! \todo rename variables here
	for(uint groupBase = 0; groupBase < s_incomingCount; groupBase += GROUP_SIZE) {

		__shared__ size_t s_groupSize;

		//! \todo perhaps do the unpacking inside the loop?
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
			s_targetBits[threadIdx.x] = incomingTargetWarps(sgin);
			s_warpAddress[threadIdx.x] = g_fcm + incomingWarpOffset(sgin) * WARP_SIZE;
			DEBUG_MSG("c%u w%u -> p%u\n", incomingWarpOffset(sgin), CURRENT_PARTITION);
		}

		__syncthreads();

		if(threadIdx.x < SCHEDULING_THREADS){

			uint group = threadIdx.x;

			/* Bit field indicating which local warps are targeted */
			uint32_t commitTargets = 0;

			/* More complex scheduling can be achieved, but this simple scheme
			 * is fast to implement */
			uint commit = 0;
			for(uint warp = 0; warp < 8; ++warp) {
				uint32_t warpTargets = s_targetBits[group * 8 + warp];
				if(warpTargets & commitTargets) {
					// conflict
					commit++;
					commitTargets = warpTargets;
				} else {
					// no conflict
					commitTargets = commitTargets | warpTargets;
				}
				s_warpCommit[group * 8 + warp] = commit;
			}
			s_warpCommitCount[group] = commit + 1;
		}
		__syncthreads();

		for(uint gwarp_base = 0; gwarp_base < s_groupSize; gwarp_base += WARPS_PER_BLOCK) {

			uint bwarp = threadIdx.x / WARP_SIZE; // warp index within a block
			uint gwarp = gwarp_base + bwarp;      // warp index within the global schedule

			bool doCommit;
			uint postsynaptic;
			float weight = 0.0f;

			synapse_t* base = s_warpAddress[gwarp] + threadIdx.x % WARP_SIZE;

			// only warps at the very end of the group are invalid here
			//! \todo could get of this conditional altogether if we set some
			//fixed (invalid) address previously.
			if(gwarp < s_groupSize) {
				postsynaptic = targetNeuron(*base);
				weight = *((float*)base + c_fcmPlaneSize);
			}

			doCommit = weight != 0.0f;

			for(uint commit=0; commit < s_warpCommitCount[gwarp_base/WARPS_PER_BLOCK]; ++commit) {

				if(doCommit && s_warpCommit[gwarp] == commit) {
					s_current[postsynaptic] += weight;
					DEBUG_MSG("c%u p?n? -> p%un%u %+f\n",
							s_cycle, CURRENT_PARTITION, postsynaptic, weight);
				}

				__syncthreads();
			}
			__syncthreads();
		}
		__syncthreads(); // to avoid overwriting s_groupSize
	}
}
