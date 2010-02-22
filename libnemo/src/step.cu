#include "cycle.cu"

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
		uint* g_outgoingCount,
		outgoing_t* g_outgoing,
		uint* g_incomingHeads,
		incoming_t* g_incoming,
		// firing stimulus
		uint32_t* g_fstim,
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

	/* The shared memory is allocated in fixed-sized blocks. During the
	 * different stages of the kernel each block may be used for different
	 * purposes. */

	/* Per-neuron buffers */
	__shared__ float s_current[MAX_PARTITION_SIZE];
	//! \todo rename to nidx_dt for consistency
	__shared__ dnidx_t s_fired[MAX_PARTITION_SIZE];

	/* Per-neuron bit-vectors. See bitvector.cu for accessors */
	__shared__ uint32_t s_N1A[MAX_PARTITION_SIZE/32];
	__shared__ uint32_t s_N1B[MAX_PARTITION_SIZE/32];

	/* Per-partition parameters */
	__shared__ uint s_partitionSize;
	__shared__ uint s_firingCount;

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

	gather(cycle, g_fcm, g_incomingHeads, g_incoming, s_current, s_N1A, s_N1B);

	SET_COUNTER(s_ccMain, 2);

	if(g_rngState != NULL && g_rngSigma != NULL) {
		thalamicInput(s_partitionSize, s_pitch32, g_rngState, g_rngSigma, s_current);
	}

	SET_COUNTER(s_ccMain, 3);

	uint32_t* s_fstim = s_N1A;
	loadFiringInput(g_fstim, s_fstim);

	fire( s_partitionSize,
			g_neuronParameters + CURRENT_PARTITION * s_pitch32,
			s_current,
			s_fstim,
			&s_firingCount,
			s_fired);

	__syncthreads();

	uint32_t* s_dfired = s_N1A;
	storeFiringOutput(s_firingCount, s_fired, s_dfired, firingOutput);

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

	SET_COUNTER(s_ccMain, 6);

	WRITE_COUNTERS(s_ccMain, g_cycleCounters, ccPitch, CC_MAIN_COUNT);
}
