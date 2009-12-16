#include "cycle.cu"

#undef STDP_FN
#ifdef STDP
#define STDP_FN(f) f ## _STDP
#else
#define STDP_FN(f) f ## _static
#endif


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
		uint64_t* g_recentFiring,
		// neuron state
		float* g_neuronParameters,
		unsigned* g_rngState,
		//! \todo combine with g_neuronParameters
		float* g_sigma,
		size_t neuronParametersSize,
		// connectivity
		uint* gf0_cm, uint64_t* gf0_delays,
		uint* gf1_cm, uint64_t* gf1_delays,
		// L1 spike queue
		uint2* gSpikeQueue,
		size_t sqPitch,
		unsigned int* gSpikeQueueHeads,
		size_t sqHeadPitch,
		// firing stimulus
		uint32_t* g_fstim,
		size_t pitch1,
#ifdef KERNEL_TIMING
		// cycle counting
		unsigned long long* g_cycleCounters,
		size_t ccPitch,
#endif
		uint32_t* firingOutput) // already offset to current cycle
{
	SET_COUNTER(s_ccMain, 0);

	/* The shared memory is allocated in fixed-sized blocks. During the
	 * different stages of the kernel each block may be used for different
	 * purposes. */

	/* Per-neuron buffers */
	__shared__ uint32_t s_M1KA[MAX_PARTITION_SIZE];
	__shared__ uint64_t s_M1KB[MAX_PARTITION_SIZE];

	/* Per-thread buffers */
	__shared__ uint16_t s_T16[THREADS_PER_BLOCK];
	__shared__ uint32_t s_T32[THREADS_PER_BLOCK];

	/* Per-delay buffer */
	__shared__ uint32_t s_D32[MAX_DELAY];

	/* Per-partition buffer */
	__shared__ uint32_t s_P32[MAX_PARTITION_COUNT];

	uint64_t* s_recentFiring = s_M1KB;

	/* Per-partition parameters */
	__shared__ uint s_partitionSize;
	__shared__ uint sf1_maxSynapsesPerDelay;
	__shared__ float s_substepMult;

	__shared__ uint32_t* s_fcmAddr[MAX_DELAY];
	__shared__ ushort2 s_fcmPitch[MAX_DELAY]; // ... and pre-computed chunk count

	if(threadIdx.x == 0) {
#ifdef __DEVICE_EMULATION__
		s_cycle = cycle;
#endif
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		sf1_maxSynapsesPerDelay = cf1_maxSynapsesPerDelay[CURRENT_PARTITION];
		s_substepMult = 1.0f / __int2float_rn(substeps);
    }
	__syncthreads();

	loadNetworkParameters();

#ifdef STDP
	loadStdpParameters();
#endif
	/* Within a connection matrix plane, partitionRow is the row offset of the
	 * current partition. The offset in /words/ differ between forward/reverse
	 * and level 0/1 as they have different row pitches */
	size_t f_partitionRow = CURRENT_PARTITION * s_maxPartitionSize * s_maxDelay;

	SET_COUNTER(s_ccMain, 1);

    //! \todo no need to clear array here, if loading thalamic input
	setSharedArray(s_M1KA, 0);
	float* s_current = (float*) s_M1KA;
    if(g_rngState != NULL && g_sigma != NULL) {
        thalamicInput(s_partitionSize,
                neuronParametersSize,
                s_pitch32,
                g_rngState,
                g_sigma,
                s_current);
    }

	SET_COUNTER(s_ccMain, 2);

	loadSharedArray(s_partitionSize,
			s_pitch64,
			g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch64,
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
				s_current,
				s_P32);
	}

	SET_COUNTER(s_ccMain, 4);

	deliverL0Spikes_(
			s_partitionSize,
			s_recentFiring,
			gf0_delays + CURRENT_PARTITION * s_pitch64,
			s_current, s_T16, s_T32, s_D32,
			s_fcmAddr, s_fcmPitch);

	SET_COUNTER(s_ccMain, 5);

	/* The dense firing output is staged in shared memory before being written
	 * to global memory */
	clearFiringOutput();

	//__shared__ uint32_t s_fstim[DIV_CEIL(STDP_FN(MAX_PARTITION_SIZE), 32)];
	//! \todo use the same buffer for both input and output
	/* Make sure s_T16 is large enough */
	uint32_t* s_fstim = (uint32_t*) s_T16;
	bool hasExternalInput = g_fstim != 0;
	ASSERT(THREADS_PER_BLOCK/2 >= DIV_CEIL(MAX_PARTITION_SIZE, 32));
	loadExternalFiring(hasExternalInput, s_partitionSize, pitch1, g_fstim, s_fstim);

	fire(
			s_partitionSize,
			substeps, s_substepMult,
			pitch1,
			g_neuronParameters + CURRENT_PARTITION * s_pitch32,
			neuronParametersSize,
			s_current, 
			s_fstim);

	__syncthreads();

	writeFiringOutput(firingOutput + CURRENT_PARTITION * pitch1, pitch1);

	SET_COUNTER(s_ccMain, 6);

#ifdef STDP
	updateSTDP_(
			false,
			s_recentFiring,
			s_recentFiring,
			s_pitch32,
			s_partitionSize,
			cr0_address, cr0_stdp, cr0_pitch,
			s_T32);
#endif
	SET_COUNTER(s_ccMain, 7);
#ifdef STDP
	if(haveL1) {
		updateSTDP_(
				true,
				g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch64,
				s_recentFiring,
				s_pitch64,
				s_partitionSize,
				cr1_address, cr1_stdp, cr1_pitch,
				s_T32);
	}
#endif
	SET_COUNTER(s_ccMain, 8);

	/* We need the (updated) recent firing history for L1 spike
	 * delivery later, but won't update this further, so we can write
	 * back to global memory now. */
	updateHistory(s_partitionSize, s_recentFiring,
			g_recentFiring
				+ writeBuffer(cycle) * PARTITION_COUNT * s_pitch64
				+ CURRENT_PARTITION * s_pitch64);
	//! \todo add an additional counter?

	if(haveL1) {
		STDP_FN(deliverL1Spikes_JIT)(
				s_maxDelay,
                writeBuffer(cycle),
				s_partitionSize,
				//! \todo need to call this differently from wrapper
				sf1_maxSynapsesPerDelay,
				gf1_cm + f_partitionRow * sf1_pitch, sf1_pitch, sf1_size,
				s_recentFiring,
				gf1_delays + CURRENT_PARTITION * s_pitch64,
				(uint2*) s_M1KA, // used for s_current previously, now use for staging outgoing spikes
				//! \todo compile-time assertions to make sure we're not overflowing here
				//! \todo fix naming!
				gSpikeQueue,
				sqPitch,
				gSpikeQueueHeads,
				sqHeadPitch,
				s_T16, s_T32, s_D32, s_P32,
				s_fcmAddr, s_fcmPitch);
	}

	SET_COUNTER(s_ccMain, 9);
	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
