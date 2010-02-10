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
		// spike delivery
		synapse_t* g_fcm,
		uint* g_outgoingCount,
		outgoing_t* g_outgoing,
		uint* g_incomingHeads,
		incoming_t* g_incoming,
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
	__shared__ uint64_t s_N64[MAX_PARTITION_SIZE];
	__shared__ dnidx_t s_fired[MAX_PARTITION_SIZE];
	__shared__ uint32_t s_N1[MAX_PARTITION_SIZE/32];

	/* Per-partition parameters */
	__shared__ uint s_partitionSize;
	__shared__ float s_substepMult;
	__shared__ uint s_firingCount;

	if(threadIdx.x == 0) {
#ifdef __DEVICE_EMULATION__
		s_cycle = cycle;
#endif
		s_firingCount = 0;
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		//! \todo no need to compute this on device.
		s_substepMult = 1.0f / __int2float_rn(substeps);
    }
	__syncthreads();

	loadNetworkParameters();

#ifdef STDP
	loadStdpParameters();
#endif

	float* s_current = (float*) s_N64;
	for(int i=0; i<DIV_CEIL(MAX_PARTITION_SIZE, THREADS_PER_BLOCK); ++i) {
		s_current[i*THREADS_PER_BLOCK + threadIdx.x] = 0.0f;

	}
	SET_COUNTER(s_ccMain, 1);

	gather(cycle, g_fcm, g_incomingHeads, g_incoming, s_current);

	SET_COUNTER(s_ccMain, 2);

	if(g_rngState != NULL && g_sigma != NULL) {
		thalamicInput(s_partitionSize, neuronParametersSize,
				s_pitch32, g_rngState, g_sigma, s_current);
	}

	SET_COUNTER(s_ccMain, 3);

	uint32_t* s_fstim = s_N1;
	bool hasExternalInput = g_fstim != 0;
	ASSERT(THREADS_PER_BLOCK/2 >= DIV_CEIL(MAX_PARTITION_SIZE, 32));
	loadExternalFiring(hasExternalInput, s_partitionSize, pitch1, g_fstim, s_fstim);

	fire( s_partitionSize,
			substeps, s_substepMult,
			pitch1,
			g_neuronParameters + CURRENT_PARTITION * s_pitch32,
			neuronParametersSize,
			s_current,
			s_fstim,
			&s_firingCount,
			s_fired);

	__syncthreads();

	uint32_t* s_dfired = s_N1;
	writeFiringOutput(
			s_firingCount,
			s_fired,
			s_dfired,
			pitch1,
			firingOutput + CURRENT_PARTITION * pitch1);

	SET_COUNTER(s_ccMain, 4);

	scatter(
			cycle,
			s_firingCount,
			s_fired,
			g_outgoingCount,
			g_outgoing,
			g_incomingHeads,
			g_incoming);

	SET_COUNTER(s_ccMain, 5);

#ifdef STDP
	uint64_t* s_recentFiring = s_N64;
	loadSharedArray(s_partitionSize,
			s_pitch64,
			g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch64,
			s_recentFiring);
	__syncthreads();

	SET_COUNTER(s_ccMain, 6);

	/*! \todo since we use the same FCM for both L0 and L1, we could
	 * potentially use a single RCM and do all STDP in one go */
	updateSTDP_(
			false,
			s_recentFiring,
			s_recentFiring,
			s_pitch32,
			s_partitionSize,
			cr0_address, cr0_stdp, cr0_pitch,
			s_fired);

	SET_COUNTER(s_ccMain, 7);

	updateSTDP_(
			true,
			g_recentFiring + readBuffer(cycle) * PARTITION_COUNT * s_pitch64,
			s_recentFiring,
			s_pitch64,
			s_partitionSize,
			cr1_address, cr1_stdp, cr1_pitch,
			s_fired);

	SET_COUNTER(s_ccMain, 8);

	updateHistory(s_partitionSize,
			s_dfired,
			s_recentFiring,
			g_recentFiring
				+ writeBuffer(cycle) * PARTITION_COUNT * s_pitch64
				+ CURRENT_PARTITION * s_pitch64);
#else
	SET_COUNTER(s_ccMain, 6);
	SET_COUNTER(s_ccMain, 7);
	SET_COUNTER(s_ccMain, 8);
#endif

	SET_COUNTER(s_ccMain, 9);

	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
