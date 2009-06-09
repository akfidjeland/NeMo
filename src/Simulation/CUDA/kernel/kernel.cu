#undef STDP_FN
#ifdef STDP
#define STDP_FN(f) f ## _STDP
#else
#define STDP_FN(f) f ## _static
#endif

#define MAX_PARTITION_SIZE_static MAX_PARTITION_SIZE



//=============================================================================
// Shared memory buffers
//=============================================================================



/* Set shared memory array to fixed value */
__device__
void
STDP_FN(setSharedArray)(uint32_t* s_mem, uint32_t val)
{
	// the compiler should unroll this
	for(int i=0; i<STDP_FN(MAX_PARTITION_SIZE)/THREADS_PER_BLOCK; ++i) {
		s_mem[i*THREADS_PER_BLOCK + threadIdx.x] = val;
	}
}



/* Load firing data for d most recent simulation cycles for all neurons */
//! \todo generalise to deal with longer delays than 32.
__device__
void
STDP_FN(loadSharedArray)(
        int s_partitionSize,
        int s_neuronsPerThread,
        size_t pitch32,
		uint32_t* g_arr,
        uint32_t* s_arr)
{
	for(int i=0; i < s_neuronsPerThread; ++i) {
		if(activeNeuron(i, s_partitionSize)){
			s_arr[threadIdx.x + i*THREADS_PER_BLOCK] =
				g_arr[mul24(blockIdx.x, pitch32) + threadIdx.x + i*THREADS_PER_BLOCK];
		}
	}
}


#ifdef STDP
#include "stdp.cu"
#endif



__device__
void
STDP_FN(fire)(
	bool hasExternalInput,
	uint s_partitionSize,
	uint s_neuronsPerThread,
	uint substeps,
	float substepMult, // substepMul * substeps = 1
	size_t fstimPitch,
	size_t pitch32,
	float* g_neuronParameters,
	size_t neuronParametersSize,
	// input
	uint32_t* g_fstim,                            // externally driven firing
	float* s_current,                              // input current
	// buffers
	uint32_t* s_recentFiring,
	uint32_t* g_recentFiring,
#ifdef STDP
	uint32_t* s_recentArrivals,
	uint32_t* g_recentArrivals,
#endif
	uint16_t* s_firingIdx,
	uint* s_nextIdxEntry)
{
	float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	float* g_u = g_neuronParameters + STATE_U * neuronParametersSize;
	float* g_v = g_neuronParameters + STATE_V * neuronParametersSize;

	/* We could save a small amount of shared memory by moving this inside the
	 * firing loop. However, as it is it's typically no more than a single
	 * warp's worth of data, so saving this shared memory will have an adverse
	 * effect on global memory usgage */
	__shared__ uint32_t s_fstim[DIV_CEIL(STDP_FN(MAX_PARTITION_SIZE), 32)];
	loadExternalFiring(hasExternalInput, s_partitionSize,
        fstimPitch, g_fstim, s_fstim);
	
	for( int i=0; i<s_neuronsPerThread; ++i ){

		if(activeNeuron(i, s_partitionSize)) {

			int s_index = i*THREADS_PER_BLOCK + threadIdx.x;
			int g_index = mul24(blockIdx.x, pitch32) + s_index;

			float v = g_v[g_index];
			float u = g_u[g_index];
			float a = g_a[g_index];
			float b = g_b[g_index];
			float I = s_current[s_index];

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
			bool forceFiring = (s_fstim[s_index/32] >> (s_index % 32)) & 0x1;
			uint32_t firing = 0;

			if(fired || forceFiring) {
                //! \todo could probably hard-code c and d 
				v = g_c[g_index];
				u += g_d[g_index];
				firing = 0x1;
				DEBUG_MSG("%d fired\n", s_index);
				int idxEntry = atomicAdd(s_nextIdxEntry, 1);
				s_firingIdx[idxEntry] = (uint16_t) s_index;
			}
			/* We need the (updated) recent firing history for L1 spike
			 * delivery later, but won't update this further, so we can write
			 * back to global memory now. */
			s_recentFiring[s_index] = (s_recentFiring[s_index] << 1) | firing;
			g_recentFiring[g_index] = s_recentFiring[s_index];
			//! \todo should we keep *updated* s_recentArrivals for LTP?
#ifdef STDP
			g_recentArrivals[g_index] = s_recentArrivals[s_index] << 1;
#endif
			g_v[g_index] = v;
			g_u[g_index] = u;
		}
	}
}


__device__
void
STDP_FN(deliverL0Spikes)(
	uint maxDelay,
	int s_partitionSize,
	size_t pitchCM, // word pitch
	int s_maxL0Synapses,
	//inputs
	uint* g_saddress,
	float* g_sweights,
	uint32_t* s_recentFiring,
#ifdef STDP
	uint32_t* s_recentIncoming,
	float* g_ltd,
	uint stdpCycle,
#endif
	uint32_t* g_firingDelays,
	float* s_current)
{
	/*! \note This is the maximum number of chunks required for this whole
	 * cluster. It should be possible to reduce this for rows with few entries.
	 * Perhaps better to just save the number of chunks in constant memory. It
	 * would depend on the chunk size, though. */
	__shared__ int s_chunkCount;
	__shared__ int s_synapsesPerDelay;
	__shared__ int s_delaysPerChunk;
	__shared__ int s_chunksPerDelay;

	if(threadIdx.x == 0) {
		//! \todo do we need to round to block size if multiple chunks per delay?
		s_synapsesPerDelay = ALIGN(s_maxL0Synapses, warpSize);
		s_chunksPerDelay = DIV_CEIL(s_synapsesPerDelay, THREADS_PER_BLOCK);
		s_delaysPerChunk = THREADS_PER_BLOCK / s_synapsesPerDelay;
	}
	__syncthreads();

	for(int preOffset=0; preOffset < s_partitionSize; preOffset += THREADS_PER_BLOCK) {

		__shared__ int s_firingCount;
		//! \todo make this a re-usable chunk of memory
		__shared__ uint16_t s_firingIdx[THREADS_PER_BLOCK];
		__shared__ uint32_t s_arrivalBits[THREADS_PER_BLOCK];

		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		//! \todo load s_recentFiring here, write result to smem array
		int candidate = preOffset + threadIdx.x;

        /* It might seem a good idea to load firing delays from global memory *
         * inside the if-clause, so as to avoid memory traffic when little
         * firing occurs.  In practice, however, this was found to increase
         * execution time (when not firing) by 68%. It's not clear why this is
        * so. */ 
		uint32_t arrivals = s_recentFiring[candidate] & g_firingDelays[candidate];
		if(arrivals && candidate < s_partitionSize) {
			int nextFree = atomicAdd(&s_firingCount, 1);
			s_firingIdx[nextFree] = candidate;
			s_arrivalBits[nextFree] = arrivals;
		}
		__syncthreads();
		/* We now have the indices of the firing of THREADS_PER_BLOCK
		 * presynaptic neurons */
		for(int i=0; i<s_firingCount; ++i) {

			int presynaptic = s_firingIdx[i];

			__shared__ uint s_delayBlocks;
			__shared__ uint32_t s_arrivals[MAX_DELAY];
			if(threadIdx.x == 0) {
				s_delayBlocks = 0;
				uint32_t arrivalBits = s_arrivalBits[i];

				while(arrivalBits) {

					int arrivalDelay = __ffs(arrivalBits) - 1;
					s_arrivals[s_delayBlocks] = arrivalDelay;
					arrivalBits &= ~(0x1 << arrivalDelay);
					s_delayBlocks += 1;
				}
				s_chunkCount = s_delaysPerChunk == 0 ?
					s_delayBlocks * s_chunksPerDelay :  // >= 1 chunk(s) per delay
					DIV_CEIL(s_delayBlocks, s_delaysPerChunk);  // multiple delays per chunk
			}
			__syncthreads();

			/* The delay pitch may vary between networks, or between partitions.
			 * Even with sequential processing of presynaptic neurons, we want to
			 * maximise parallel processing of incoming spikes from different
			 * delays. We have two situations: 
			 *
			 * 1) if the delay pitch is more than half the block size we process
			 *    each delay sequentially
			 * 2) if the delay pitch is less than or equal to half the block size
			 *    we process multiple delays in parallel 
			 */

			for(int chunk=0; chunk < s_chunkCount; ++chunk) {

				int delayEntry = s_delaysPerChunk == 0 ?
					chunk / s_chunksPerDelay :
					threadIdx.x / s_synapsesPerDelay;
				uint32_t delay = s_arrivals[delayEntry];
				/* Offset /within/ a delay block */
				int synapseIdx = s_delaysPerChunk == 0 ?
					(chunk % s_chunksPerDelay) * THREADS_PER_BLOCK + threadIdx.x :
					(threadIdx.x % s_synapsesPerDelay);

                float weight;
                uint postsynaptic;
                bool doCommit = false;

				//! \todo consider using per-neuron maximum here instead
				if(synapseIdx < s_maxL0Synapses && delayEntry < s_delayBlocks
#ifdef __DEVICE_EMULATION__	
						// warp size is 1, so rounding to warp size not as expected
						&& threadIdx.x < s_synapsesPerDelay * s_delaysPerChunk
#endif
				) {

					size_t synapseAddress = presynaptic * maxDelay * pitchCM
					                  + delay * pitchCM
					                  + synapseIdx;
					weight = g_sweights[synapseAddress];
					uint sdata = g_saddress[synapseAddress];
					postsynaptic = targetNeuron(sdata);

					if(weight != 0.0f) {
                        doCommit = true;
#ifdef STDP
						g_saddress[synapseAddress] = setTimestamp(sdata, stdpCycle);
						//! \todo check for off-by-one errors here
						int dt = __ffs(s_recentFiring[postsynaptic]);

						/*! \todo perhaps we should only apply depression once,
						 * for first incoming of postsynaptic firing? */
						//! \todo make sure we only modify excitatory
						if(s_recentFiring[postsynaptic] && abs(dt) < s_stdpTauD) {
							g_ltd[synapseAddress] -= depression(dt);
							DEBUG_MSG("ltd: %+f for synapse %u -> %u after delay of %u\n",
								depression(dt), presynaptic, postsynaptic, dt);
						}
#endif
					}
				}

                /* Only deal with a single delay at a time, to avoid race
                 * condition resulting from multiple synapses terminating at
                 * the same postsynaptic neuron. Within a single delay, there
                 * should be no race conditions, if the mapper has done its job
                 * 
                 * It's possible to do this using mutexes based on shared
                 * memory atomics. However, this was found to be slightly
                 * slower (12% overhead vs 10% overhead wrt no work-around for
                 * race condition) and makes use of a sizable amount of
                 * precious shared memory for the mutex data.  */

                /*! \todo we can increase the amount of parallel execution here
                 * by identifying (at map time) delay blocks which have no
                 * potential race conditions. We can thus assign a "commit
                 * number" to each delay block and do these in parallel */
                for(int commit=0; commit < s_delayBlocks; ++commit) {
                    if(delayEntry == commit && doCommit) {
                        s_current[postsynaptic] += weight; 
						DEBUG_MSG("L0 current %f for synapse %u -> %u after delay %d" 
                                " (thread %u, i=%d, chunk=%d)\n",
							    weight, presynaptic, postsynaptic, 
                                delay, threadIdx.x, i, chunk);
#ifdef STDP
						s_recentIncoming[postsynaptic] |= 0x1;
#endif
                    }
                    __syncthreads();
                }
                /* We need a barrier *outside* the loop to avoid threads
                 * reaching the barrier (inside the loop), then having thread 0
                 * race ahead and changing s_delayBlocks before all threads
                 * have left the loop. */
                __syncthreads();
			}
		}
	}
}




/*! Combined integrate and fire using sparse connectivity matrix, a single step
* updates the state (u and v) of each neuron and produces spikes to be used in
* the next simulation cycle. 
* 
* The number of neurons per block provided to the kernel is always
* warp-aligned. This means that some threads do useless work, but at no cost.
* Using a warp-aligned neuron number simplifies the control when the number of
* neurons is not an exact multiple of the number of threads per block.
*
 * The parameters (a, b, c, and d) can be set for each individual neuron and
 * should be pre-loaded in global memory before this kernel is invoked.
 */
__global__
void
STDP_FN(step) (
		int substeps,
        uint32_t cycle,
		uint32_t* g_recentFiring, 
#ifdef STDP
		uint32_t* g_recentArrivals,
		uint stdpCycle,
		uint* g_cm0R,
		uint32_t* g_arrivalDelaysL0,
		uint32_t* g_arrivalDelaysL1,
#endif
		// neuron state
		float* g_neuronParameters,
        unsigned* g_rngState,
        float* g_sigma,
		size_t neuronParametersSize,
        // L0 synapses
		uint* g_L0CM,
		uint32_t* g_firingDelaysL0,
		// L1 connectivity matrix
		uint* g_L1CM,
		uint32_t* g_firingDelaysL1,
		// L1 spike queue
		uint2* gSpikeQueue, 
		size_t sqPitch, 
        unsigned int* gSpikeQueueHeads, 
        size_t sqHeadPitch,
        // firing stimulus
		uint32_t* g_fstim,
		size_t fstimPitch,
		// cycle counting
#ifdef KERNEL_TIMING
		unsigned long long* g_cycleCounters,
		size_t ccPitch,
#endif
		// firing output
		int fmemCycle,
		ushort2* g_fmemBuffer,
		uint* g_fmemNextFree)
{
	SET_COUNTER(s_ccMain, 0);

	/* The shared memory is allocated in fixed-sized blocks. During the
	 * different stages of the kernel each block may be used for different
	 * purposes. */

	__shared__ uint32_t s_M1KA[STDP_FN(MAX_PARTITION_SIZE)];
	__shared__ uint32_t s_M1KB[STDP_FN(MAX_PARTITION_SIZE)];
	//__shared__ uint32_t s_M1KC[STDP_FN(MAX_PARTITION_SIZE)];
	__shared__ uint16_t s_M512[STDP_FN(MAX_PARTITION_SIZE)];

	/* The above memory allocation leaves slightly less than 1kB for kernel
	 * parameters, individual shared variables etc */

	uint32_t* s_recentFiring = s_M1KB;
	uint16_t* s_firingIdx = s_M512;
	__shared__ uint s_firingCount;

#ifdef STDP
	__shared__ uint32_t s_recentArrivals[STDP_FN(MAX_PARTITION_SIZE)];
#endif

	/* Per-partition parameters */
	__shared__ uint s_partitionSize;
	__shared__ uint s_neuronsPerThread;
	__shared__ uint s_maxL0SynapsesPerDelay;
	__shared__ uint s_maxL1SynapsesPerDelay;
#ifdef STDP
	__shared__ uint s_maxL0RevSynapsesPerDelay;
#endif
	__shared__ float s_substepMult;

	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		s_neuronsPerThread = DIV_CEIL(s_partitionSize, THREADS_PER_BLOCK);
		s_maxL0SynapsesPerDelay = c_maxL0SynapsesPerDelay[CURRENT_PARTITION];
		s_maxL1SynapsesPerDelay = c_maxL1SynapsesPerDelay[CURRENT_PARTITION];
#ifdef STDP
		s_maxL0RevSynapsesPerDelay = c_maxL0RevSynapsesPerDelay[CURRENT_PARTITION];
#endif
		s_substepMult = 1.0f / __int2float_rn(substeps);
    }
	__syncthreads();

	loadNetworkParameters();

#ifdef STDP
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, s_pitch32, g_recentArrivals, s_recentArrivals);
	loadStdpParameters();
#endif
	SET_COUNTER(s_ccMain, 1);

    //! \todo no need to clear array here, if loading thalamic input
	STDP_FN(setSharedArray)(s_M1KA, 0);
	float* s_current = (float*) s_M1KA;
    if(g_rngState != NULL && g_sigma != NULL) {
        thalamicInput(s_partitionSize,
                s_neuronsPerThread,
                neuronParametersSize,
                s_pitch32,
                g_rngState,
                g_sigma,
                s_current);
    }

	SET_COUNTER(s_ccMain, 2);

	gatherL1Spikes_JIT(
        readBuffer(cycle),
		gSpikeQueue,
		sqPitch,
		gSpikeQueueHeads,
		sqHeadPitch,
		s_current,
#if MAX_THREAD_BLOCKS > STDP_FN(MAX_PARTITION_SIZE)
#error "Need to use larger memory buffer for spike queue heads"
#else
        s_M1KB); // only part of s_M1KB is used
#endif
    __syncthreads();


	SET_COUNTER(s_ccMain, 3);

	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, s_pitch32, g_recentFiring, s_recentFiring);
	__syncthreads();

	SET_COUNTER(s_ccMain, 4);

	STDP_FN(deliverL0Spikes)(
			s_maxDelay,
			s_partitionSize,
			s_pitchL0,
			s_maxL0SynapsesPerDelay,
			//! \todo move addressing inside function
			g_L0CM
				+ CM_ADDRESS * s_sizeL0
				+ CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_pitchL0,
			(float*) g_L0CM
				+ CM_WEIGHT * s_sizeL0
				+ CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_pitchL0,
			s_recentFiring,
#ifdef STDP
			s_recentArrivals,
			(float*) g_L0CM
				+ CM_LTD * s_sizeL0
				+ CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_pitchL0,
			stdpCycle,
#endif
			g_firingDelaysL0 + CURRENT_PARTITION * s_pitch32,
			s_current);
	__syncthreads();

	SET_COUNTER(s_ccMain, 5);

	/* We now repurpose s_firingIdx to contain the indices of the neurons which
	 * fired just now, rather than the neurons which fired and the past and
	 * whose spikes are only now reaching. It's ok to leave the existing
	 * garbage, as we keep track of the end of the new valid firing.
	 *
	 * We likewise repurpose the firing count variable */
	if(threadIdx.x == 0) {
		s_firingCount = 0;
	}
	__syncthreads();

//! \todo use more sensible CPP macros here, to postfix functions
	STDP_FN(fire)(
			g_fstim != 0,
            s_partitionSize,
            s_neuronsPerThread,
			substeps, s_substepMult,
			fstimPitch, s_pitch32, 
			g_neuronParameters,
			neuronParametersSize,
			g_fstim, s_current, 
			s_recentFiring, 
			g_recentFiring, 
#ifdef STDP
			s_recentArrivals,
			g_recentArrivals,
#endif
			s_firingIdx,
			&s_firingCount);
	__syncthreads();

	SET_COUNTER(s_ccMain, 6);
#ifdef STDP
	updateLTP(
		s_maxDelay,
		stdpCycle,
		s_maxL0RevSynapsesPerDelay,
		// reverse matrix
		g_cm0R + CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_rpitchL0,
		s_rpitchL0, s_rsizeL0,
		g_L0CM, s_pitchL0, s_sizeL0,
		s_firingIdx,
		s_firingCount,
		s_recentArrivals,
		g_arrivalDelaysL0);
	__syncthreads();
#endif
	SET_COUNTER(s_ccMain, 7);

	writeFiringOutput(fmemCycle, g_fmemNextFree, 
			s_firingIdx, s_firingCount, g_fmemBuffer);
	__syncthreads();
	SET_COUNTER(s_ccMain, 8);

	if(gSpikeQueue) {
		deliverL1Spikes_JIT(
				s_maxDelay,
                writeBuffer(cycle),
				s_partitionSize,
				//! \todo need to call this differently from wrapper
				s_pitchL1,
				s_maxL1SynapsesPerDelay,
				g_L1CM
				+ CM_ADDRESS * s_sizeL1
				+ CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_pitchL1,
				(float*) g_L1CM
				+ CM_WEIGHT * s_sizeL1
				+ CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay * s_pitchL1,
				s_recentFiring,
				//! \todo STDP
				g_firingDelaysL1 + CURRENT_PARTITION * s_pitch32,
				(uint2*) s_M1KA, // use for s_current previously, now use for staging outgoing spikes
				// buffers for write buffer and global heads
				//! \todo compile-time assertions to make sure we're not overflowing here
				//s_M512,
				//s_M512+256,
				//! \todo fix naming!
				gSpikeQueue,
				sqPitch,
				gSpikeQueueHeads,
				sqHeadPitch);
	}

	SET_COUNTER(s_ccMain, 9);
	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
