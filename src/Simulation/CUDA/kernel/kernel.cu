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
	for(int i=0; i<DIV_CEIL(STDP_FN(MAX_PARTITION_SIZE), THREADS_PER_BLOCK); ++i) {
		s_mem[i*THREADS_PER_BLOCK + threadIdx.x] = val;
	}
}





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
	//! \todo make  this reuasable memory
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
			g_v[g_index] = v;
			g_u[g_index] = u;
		}
	}
}



/* Add current to current accumulation buffer
 *
 * Since multiple spikes may terminate at the same postsynaptic neuron, some
 * care must be taken to avoid a race condition in the current update.
 *
 * We only deal with a single delay at a time, to avoid race condition
 * resulting from multiple synapses terminating at the same postsynaptic
 * neuron. Within a single delay, there should be no race conditions, if the
 * mapper has done its job
 *
 * It's possible to do this using mutexes based on shared memory atomics.
 * However, this was found to be slightly slower (12% overhead vs 10% overhead
 * wrt no work-around for race condition) and makes use of a sizable amount of
 * precious shared memory for the mutex data.  */

/*! \todo we can increase the amount of parallel execution here by identifying
 * (at map time) delay blocks which have no potential race conditions. We can
 * thus assign a "commit number" to each delay block and do these in parallel */
__device__
void
STDP_FN(commitCurrent_)(
		bool doCommit,
		uint delayEntry,
		uint s_delayBlocks,
		uint presynaptic,
		uint postsynaptic,
		float weight,
		float* s_current)
{
	for(uint commit=0; commit < s_delayBlocks; ++commit) {
		if(delayEntry == commit && doCommit) {
			s_current[postsynaptic] += weight; 
			DEBUG_MSG("L0 current %f for synapse %u -> %u\n" ,
					weight, presynaptic, postsynaptic);
		}
		__syncthreads();
	}
}



__device__
void
STDP_FN(deliverL0Spikes_)(
	uint maxDelay,
	uint partitionSize,
	uint sf0_maxSynapses,
	uint* gf0_cm, uint f0_pitch, uint f0_size,
	uint32_t* s_recentFiring,
	uint32_t* g_firingDelays,
	float* s_current,
	uint16_t* s_firingIdx,
	uint32_t* s_arrivalBits,
	uint32_t* s_arrivals)
{
	uint*  gf0_address =          gf0_cm + FCM_ADDRESS  * f0_size;
	float* gf0_weight  = (float*) gf0_cm + FCM_WEIGHT   * f0_size;

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
		s_synapsesPerDelay = ALIGN(sf0_maxSynapses, warpSize);
		s_chunksPerDelay = DIV_CEIL(s_synapsesPerDelay, THREADS_PER_BLOCK);
		s_delaysPerChunk = THREADS_PER_BLOCK / s_synapsesPerDelay;
	}
	__syncthreads();

	for(int preOffset=0; preOffset < partitionSize; preOffset += THREADS_PER_BLOCK) {

		__shared__ int s_firingCount;
		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		int candidate = preOffset + threadIdx.x;

		/* It might seem a good idea to load firing delays from global memory
		 * inside the if-clause, so as to avoid memory traffic when little
		 * firing occurs.  In practice, however, this was found to increase
		 * execution time (when not firing) by 68%. It's not clear why this is
		 * so. */ 
		uint32_t arrivals = s_recentFiring[candidate] & g_firingDelays[candidate];
		if(arrivals && candidate < partitionSize) {
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
			if(threadIdx.x == 0) {
				s_delayBlocks = 0;
				uint32_t arrivalBits = s_arrivalBits[i];

				//! \todo can do this in parallel?
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
			for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

				uint delayEntry = s_delaysPerChunk == 0 ?
					chunk / s_chunksPerDelay :
					threadIdx.x / s_synapsesPerDelay;
				uint32_t delay = s_arrivals[delayEntry];
				/* Offset /within/ a delay block */
				uint synapseIdx = s_delaysPerChunk == 0 ?
					(chunk % s_chunksPerDelay) * THREADS_PER_BLOCK + threadIdx.x :
					(threadIdx.x % s_synapsesPerDelay);

				float weight;
				uint postsynaptic;
				bool doCommit = false;

				//! \todo consider using per-neuron maximum here instead
				if(synapseIdx < sf0_maxSynapses && delayEntry < s_delayBlocks
#ifdef __DEVICE_EMULATION__	
						// warp size is 1, so rounding to warp size not as expected
						&& threadIdx.x < s_synapsesPerDelay * s_delaysPerChunk
#endif
				) {

					size_t synapseAddress = 
						(presynaptic * maxDelay + delay) * f0_pitch + synapseIdx;
					weight = gf0_weight[synapseAddress];

					/*! \todo only load address if it will actually be used.
					 * For benchmarks this made little difference, presumably
					 * because all neurons have same number of synapses.
					 * Experiment! */
					uint sdata = gf0_address[synapseAddress];
					postsynaptic = targetNeuron(sdata);

					doCommit = weight != 0.0f;
				}

				STDP_FN(commitCurrent_)(doCommit, delayEntry, s_delayBlocks, 
						presynaptic, postsynaptic, weight, s_current);

				/* We need a barrier *outside* the loop to avoid threads
				 * reaching the barrier (inside the loop), then having thread 0
				 * race ahead and changing s_delayBlocks before all threads
				 * have left the loop. */
				__syncthreads();
			}
		}
	}
	__syncthreads();
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
		// neuron state
		float* g_neuronParameters,
        unsigned* g_rngState,
		//! \todo combine with g_neuronParameters
        float* g_sigma,
		size_t neuronParametersSize,
		// connectivity
		uint* gf0_cm, uint32_t* gf0_delays,
		uint* gf1_cm, uint32_t* gf1_delays,
#ifdef STDP
		uint* gr0_cm,
		uint* gr1_cm,
#endif
		// L1 spike queue
		uint2* gSpikeQueue, 
		size_t sqPitch, 
        unsigned int* gSpikeQueueHeads, 
        size_t sqHeadPitch,
        // firing stimulus
		uint32_t* g_fstim,
		size_t fstimPitch,
#ifdef KERNEL_TIMING
		// cycle counting
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

	/* Per-neuron buffers */
	__shared__ uint32_t s_M1KA[STDP_FN(MAX_PARTITION_SIZE)];
	__shared__ uint32_t s_M1KB[STDP_FN(MAX_PARTITION_SIZE)];
	__shared__ uint16_t s_M512[STDP_FN(MAX_PARTITION_SIZE)];

	/* Per-thread buffers */
	__shared__ uint16_t s_T16[THREADS_PER_BLOCK];
	__shared__ uint32_t s_T32[THREADS_PER_BLOCK];

	/* Per-delay buffers */
	__shared__ uint32_t s_D32[MAX_DELAY];

	uint32_t* s_recentFiring = s_M1KB;
	//! \todo we could probably get away with per-thread storage here, by reorganising kernel
	uint16_t* s_firingIdx = s_M512;

	/* Per-partition parameters */
	__shared__ uint s_partitionSize;
	__shared__ uint s_neuronsPerThread;
	__shared__ uint sf0_maxSynapsesPerDelay;
	__shared__ uint sf1_maxSynapsesPerDelay;
#ifdef STDP
	__shared__ uint sr0_maxSynapsesPerNeuron;
	__shared__ uint sr1_maxSynapsesPerNeuron;
#endif
	__shared__ float s_substepMult;

	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		s_neuronsPerThread = DIV_CEIL(s_partitionSize, THREADS_PER_BLOCK);
		sf0_maxSynapsesPerDelay = cf0_maxSynapsesPerDelay[CURRENT_PARTITION];
		sf1_maxSynapsesPerDelay = cf1_maxSynapsesPerDelay[CURRENT_PARTITION];
#ifdef STDP
		sr0_maxSynapsesPerNeuron = cr0_maxSynapsesPerNeuron[CURRENT_PARTITION];
		sr1_maxSynapsesPerNeuron = cr1_maxSynapsesPerNeuron[CURRENT_PARTITION];
#endif
		s_substepMult = 1.0f / __int2float_rn(substeps);
    }
	__syncthreads();

	loadNetworkParameters();

#ifdef STDP
	loadStdpParameters();

	/* The reverse matrix uses one row per neuron rather than per delay.  The
	 * word offset may differ in levels 0 and 1. */
	size_t r_partitionRow = CURRENT_PARTITION * s_maxPartitionSize;
#endif
	/* Within a connection matrix plane, partitionRow is the row offset of the
	 * current partition. The offset in /words/ differ between forward/reverse
	 * and level 0/1 as they have different row pitches */
	size_t f_partitionRow = CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay;

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

	loadSharedArray(s_partitionSize, s_neuronsPerThread,
			s_pitch32,
			g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch32,
			s_recentFiring);
	__syncthreads();

	SET_COUNTER(s_ccMain, 3);

	bool haveL1 = gSpikeQueue != NULL;
	if(haveL1) {
		STDP_FN(gatherL1Spikes_JIT_)(
				readBuffer(cycle),
				gSpikeQueue,
				sqPitch,
				gSpikeQueueHeads,
				sqHeadPitch,
				s_current);
	}

	SET_COUNTER(s_ccMain, 4);

	STDP_FN(deliverL0Spikes_)(
			s_maxDelay,
			s_partitionSize,
			sf0_maxSynapsesPerDelay,
			gf0_cm + f_partitionRow * sf0_pitch, sf0_pitch, sf0_size,
			s_recentFiring,
			gf0_delays + CURRENT_PARTITION * s_pitch32,
			s_current, s_T16, s_T32, s_D32);

	SET_COUNTER(s_ccMain, 5);

	__shared__ uint s_firingCount;
	if(threadIdx.x == 0) {
		s_firingCount = 0;
	}
	__syncthreads();

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
			g_recentFiring + writeBuffer(cycle) * PARTITION_COUNT * s_pitch32,
			s_firingIdx,
			&s_firingCount);
	__syncthreads();
	SET_COUNTER(s_ccMain, 6);

#ifdef STDP
	updateSTDP_(
			false,
			s_recentFiring,
			s_recentFiring,
			s_pitch32, 1,
			s_partitionSize,
			sr0_maxSynapsesPerNeuron,
			gr0_cm + r_partitionRow * sr0_pitch, sr0_pitch, sr0_size,
			s_T32);
#endif
	SET_COUNTER(s_ccMain, 7);
#ifdef STDP
	if(haveL1) {
		updateSTDP_(
				true,
				s_recentFiring,
				g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch32,
				s_pitch32, 0,
				s_partitionSize,
				sr1_maxSynapsesPerNeuron,
				gr1_cm + r_partitionRow * sr1_pitch, sr1_pitch, sr1_size,
				s_T32);
	}
#endif
	SET_COUNTER(s_ccMain, 8);

	writeFiringOutput(fmemCycle, g_fmemNextFree, 
			s_firingIdx, s_firingCount, g_fmemBuffer);
	__syncthreads();
	SET_COUNTER(s_ccMain, 9);

	if(haveL1) {
		STDP_FN(deliverL1Spikes_JIT)(
				s_maxDelay,
                writeBuffer(cycle),
				s_partitionSize,
				//! \todo need to call this differently from wrapper
				sf1_maxSynapsesPerDelay,
				gf1_cm + f_partitionRow * sf1_pitch, sf1_pitch, sf1_size,
				s_recentFiring,
				gf1_delays + CURRENT_PARTITION * s_pitch32,
				(uint2*) s_M1KA, // used for s_current previously, now use for staging outgoing spikes
				//! \todo compile-time assertions to make sure we're not overflowing here
				//! \todo fix naming!
				gSpikeQueue,
				sqPitch,
				gSpikeQueueHeads,
				sqHeadPitch,
				s_T16, s_T32, s_D32);
	}

	SET_COUNTER(s_ccMain, 10);
	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}

