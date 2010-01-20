#include "cycle.cu"
#include "dispatchTable.cu"

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



/* Set shared memory array to fixed value */
__device__
void
setSharedArray(uint32_t* s_mem, uint32_t val)
{
	// the compiler should unroll this
	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		s_mem[i*THREADS_PER_BLOCK + threadIdx.x] = val;
	}
}



__device__
void
updateHistory(uint s_partitionSize, uint64_t* s_recentFiring, uint64_t* g_recentFiring)
{
	for(uint nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {
		uint neuron = nbase + threadIdx.x;
		if(neuron < s_partitionSize) {
			/* Need to update firing history here as we need it in L1 delivery,
			 * so we can handle 1-cycle delay */
			s_recentFiring[neuron] = (s_recentFiring[neuron] << 1) | (didFire(neuron) ? 0x1 : 0x0);
			g_recentFiring[neuron] = s_recentFiring[neuron];
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
	uint32_t* s_fstim)   // s_T16, so larger than needed
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
				setFiringOutput(neuron);
			}

			g_v[neuron] = v;
			g_u[neuron] = u;
		}
	}
}



/* TODO: the loop structure here is nearly the same as deliverL0Spikes. Factor
 * out or use a code generator to avoid repetition */
__device__
void
l1scatter(
		uint cycle,
		uint partitionSize,
		uint64_t* s_recentFiring,
		uint16_t* s_firingIdx,
		uint* g_outgoingCount,
		outgoing_t* g_outgoing,
		uint* g_incomingHeads,
		incoming_t* g_incoming)
{
	for(int preOffset=0; preOffset < partitionSize; preOffset += THREADS_PER_BLOCK) {

		__shared__ uint s_firingCount;

		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		/*! \todo merge this step with the dumping to firing output. We can
		 * then get rid of the whole outer loop here */
		int candidate = preOffset + threadIdx.x;
		if(s_recentFiring[candidate] & 0x1) {
			int nextFree = atomicAdd(&s_firingCount, 1);
			s_firingIdx[nextFree] = candidate;
		}
		__syncthreads();

		//! \todo pre-load the outgoing count for each firing neuron (s_len and s_blocks)

		/* We now have the indices of the firing of THREADS_PER_BLOCK
		 * presynaptic neurons */
		for(uint i=0; i<s_firingCount; ++i) {

			int presynaptic = s_firingIdx[i];

			__shared__ uint s_len;
			__shared__ uint s_blocks;
			if(threadIdx.x == 0) {
				s_len = outgoingCount(presynaptic, g_outgoingCount);
				s_blocks = DIV_CEIL(s_len, THREADS_PER_BLOCK);
			}
			__syncthreads();

			for(uint block = 0; block < s_blocks; ++block) {

				uint jobIdx = block * THREADS_PER_BLOCK + threadIdx.x;

				if(jobIdx < s_len) {

					outgoing_t sout = outgoing(presynaptic, jobIdx, g_outgoing);

					uint delay = outgoingDelay(sout);

					ASSERT(delay > 0);

					uint targetPartition = outgoingTargetPartition(sout);
					size_t headsAddr = incomingCountAddr(targetPartition, cycle, delay);
					uint offset = atomicAdd(g_incomingHeads + headsAddr, 1);

					ASSERT(offset < c_incomingPitch);

					size_t base = incomingBufferStart(targetPartition, cycle, delay);
					g_incoming[base + offset] =
						make_incoming(CURRENT_PARTITION, presynaptic,
								delay,
								outgoingWarps(sout));

					DEBUG_MSG("c%u spike group p%un%u -> p%u (delay %u) (buffer entry %u/%u)\n",
							cycle, CURRENT_PARTITION, presynaptic, targetPartition, delay, offset, c_incomingPitch);
				}
			}
			__syncthreads(); // so s_blocks is not updated
		}
	}
}


#ifdef __DEVICE_EMULATION__
//! \todo remove debugging code
__shared__ uint s_valid[8];
__shared__ uint s_invalid[8];
#endif



__device__
void
l1gather(
		uint cycle,
		uint* g_incomingCount,
		incoming_t* g_incoming,
		uint16_t s_sourceNeuron[],
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
	 * memory. Doubling the group size (to 2*MAX_DELAY) did not have any
	 * measurable effect on performance, however. */
#define GROUP_SIZE (2*MAX_DELAY)

	//! \todo could this smem be re-used?
	//__shared__ uint s_synapseCount[GROUP_SIZE]; // for each incoming group
	__shared__ uint s_warpCount[GROUP_SIZE]; // for each incoming group
	__shared__ uint32_t* sf_cm[GROUP_SIZE];
	__shared__ ushort2 sf_pitch[GROUP_SIZE];

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

		//! \todo factor this out
		if(threadIdx.x < s_groupSize) {
			incoming_t sgin = getIncoming(cycle, group, g_incoming);
			uint delay = incomingDelay(sgin);
			s_sourceNeuron[threadIdx.x] = incomingNeuron(sgin);
			//s_synapseCount[threadIdx.x] = incomingWarps(sgin) * WARP_SIZE;
			s_warpCount[threadIdx.x] = incomingWarps(sgin);
			uint sourcePartition = incomingPartition(sgin);
			fcm_ref_t fcm = getFCM(sourcePartition, CURRENT_PARTITION, delay-1);
			sf_cm[threadIdx.x] = f_base(fcm);
			ASSERT(sf_cm[threadIdx.x] != 0x0);
			sf_pitch[threadIdx.x].x = f_pitch(fcm);
			//! \todo perhaps this is not needed at all?
			sf_pitch[threadIdx.x].y = DIV_CEIL(s_warpCount[threadIdx.x] * WARP_SIZE, THREADS_PER_BLOCK);
			DEBUG_MSG("c%u incoming spike group p%u -> p%u (delay %u) (%u synapses, %u chunks)\n",
					cycle, sourcePartition, CURRENT_PARTITION, delay,
					sf_pitch[threadIdx.x].x,
					sf_pitch[threadIdx.x].y);
		}

		__syncthreads();

		for(uint groupOffset = 0; groupOffset < s_groupSize; ++groupOffset) {

			for(uint chunk = 0; chunk < sf_pitch[groupOffset].y; ++chunk) {

#ifdef __DEVICE_EMULATION__
				if(threadIdx.x == 0) {
					DEBUG_MSG("c%u group %u chunk %u %u (warp-aligned) synapses (out of 256). %u synapses vs %u pitch\n",
							cycle, groupOffset, chunk, s_synapseCount[groupOffset],
							s_synapseCount[groupOffset], sf_pitch[groupOffset].x);

				}
#endif
				uint synapseIdx = chunk * THREADS_PER_BLOCK + threadIdx.x;
				bool doCommit = false;
#ifdef __DEVICE_EMULATION__
				uint presynaptic = s_sourceNeuron[groupOffset];
#endif
				uint postsynaptic;
				float weight;

				if(synapseIdx < s_warpCount[groupOffset] * WARP_SIZE) {

					size_t synapseAddress = f_synapseOffset(s_sourceNeuron[groupOffset], sf_pitch[groupOffset].x, synapseIdx);
					uint* gf0_address = f_address(sf_cm[groupOffset], sf_pitch[groupOffset].x);

					float* gf0_weight = f_weights(sf_cm[groupOffset], sf_pitch[groupOffset].x);
					weight = gf0_weight[synapseAddress];
					doCommit = weight != 0.0f;

					/*! \todo only load address if it will actually be used.  For benchmarks
					 * this made little difference, presumably because all neurons have same
					 * number of synapses.  Experiment! */
					uint sdata = gf0_address[synapseAddress];
					postsynaptic = targetNeuron(sdata);
				}

				/* Since multiple spikes may terminate at the same postsynaptic
				 * neuron, some care must be taken to avoid a race condition in
				 * the current update.
				 *
				 * Since we only deal with a single delay at a time, there
				 * should be no race conditions resulting from multiple
				 * synapses terminating at the same postsynaptic neuron.
				 * Within a single delay, there should be no race conditions,
				 * if the mapper has done its job */

				uint warp = threadIdx.x / 32;

				for(uint commit=0; commit < 8; ++commit) {

					if(doCommit && warp == commit) {
						s_current[postsynaptic] += weight;
						//! \todo add partition numbers here as well. This is not only for L1 any more
						DEBUG_MSG("c%u L0 n%u -> n%u %+f\n",
								s_cycle, presynaptic, postsynaptic, weight);
					}

#ifdef __DEVICE_EMULATION__
					if(doCommit) {
						atomicAdd(&s_valid[warp], 1);
					} else {
						atomicAdd(&s_invalid[warp], 1);
					}
#endif
					__syncthreads();
				}
			}
			__syncthreads();
		}
		__syncthreads(); // to avoid overwriting s_groupSize
	}
}
