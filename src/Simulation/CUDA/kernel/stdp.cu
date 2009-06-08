/*! \todo could use a constant+shared memory lookup table for this instead of
 * computing exponentials */

#include "log.hpp"

/* STDP parameters
 *
 * The STDP parameters apply to all neurons in a network. One might, however,
 * want to do this differently for different neuron populations. This is not
 * yet supported.
 *
 * The STDP parameters are stored in constant memory as we're running out of
 * available kernel paramters.
 *
 * We postfix parameters either P or D to indicate whether the parameter refers
 * to potentiation or depression.
 *
 * - tau specifies the maximum delay between presynaptic spike and
 *   postsynaptic firing for which STDP has an effect.
 * - alpha is a multiplier for the exponential
 */

__constant__ int c_stdpTauP;
__constant__ int c_stdpTauD;
__constant__ float c_stdpTauInvP;
__constant__ float c_stdpTauInvD;
__constant__ float c_stdpAlphaP;
__constant__ float c_stdpAlphaD;


#define SET_STDP_PARAMETER(symbol, val) CUDA_SAFE_CALL(\
        cudaMemcpyToSymbol(symbol, &val, sizeof(val), 0, cudaMemcpyHostToDevice)\
    )

__host__
void
configureStdp(int tauP, int tauD, float alphaP, float alphaD)
{
    float tauInvP = (float) (1.0 / (double) tauP);
    float tauInvD = (float) (1.0 / (double) tauD);
    SET_STDP_PARAMETER(c_stdpTauP, tauP);
    SET_STDP_PARAMETER(c_stdpTauD, tauD);
    SET_STDP_PARAMETER(c_stdpTauInvP, tauInvP);
    SET_STDP_PARAMETER(c_stdpTauInvD, tauInvD);
    SET_STDP_PARAMETER(c_stdpAlphaP, alphaP);
    SET_STDP_PARAMETER(c_stdpAlphaD, alphaD);
}


/* In the kernel we load the parameters into shared memory. These variables can
 * then be accessed using broadcast */

__shared__ int s_stdpTauP;
__shared__ int s_stdpTauD;
__shared__ float s_stdpTauInvP;
__shared__ float s_stdpTauInvD;
__shared__ float s_stdpAlphaP;
__shared__ float s_stdpAlphaD;


#define LOAD_STDP_PARAMETER(symbol) s_ ## symbol = c_ ## symbol

__device__
void
loadStdpParameters()
{
    //! \todo could use an array for this and load in parallel
    if(threadIdx.x == 0) {
        LOAD_STDP_PARAMETER(stdpTauP);
        LOAD_STDP_PARAMETER(stdpTauD);
        LOAD_STDP_PARAMETER(stdpTauInvP);
        LOAD_STDP_PARAMETER(stdpTauInvD);
        LOAD_STDP_PARAMETER(stdpAlphaP);
        LOAD_STDP_PARAMETER(stdpAlphaD);
    }
    __syncthreads();
}



__device__
float
depression(int dt)
{
	return s_stdpAlphaD * exp(__int2float_rn(-dt)*s_stdpTauInvD);
}


__device__
float
potentiation(int dt)
{
	return s_stdpAlphaP * exp(__int2float_rn(-dt)*s_stdpTauInvP);
}


/*! Process each firing neuron, potentiating synapses with spikes reaching the
 * fired neuron shortly before firing. */
__device__
void
updateLTP(
	uint maxDelay,
	uint currentTime,
	uint s_maxL0SynapsesR,
	// reverse connectivity
	uint* g_cmR, size_t cmPitchR, size_t cmSizeR,
	// forward connectivity
	uint* g_cmF, size_t cmPitchF, size_t cmSizeF,
	uint16_t* s_firingIdx,
	//! \todo change to uint?
	int s_firingCount,
	uint32_t* s_recentArrivals,
	uint32_t* g_arrivalDelays)
{
	/*! \note This is the maximum number of chunks required for this whole
	 * cluster. It should be possible to reduce this for rows with few
	 * entries. Perhaps better to just save the number of chunks in
	 * constant memory. It would depend on the chunk size, though. */
	//! \todo change to uint
	__shared__ int s_chunkCount;
	__shared__ int s_synapsesPerDelay;
	__shared__ int s_delaysPerChunk;
	__shared__ int s_chunksPerDelay;

	float* g_ltp = (float*) (g_cmR + RCM_LTP * cmSizeR);

	//! \todo factor this out and share with integrate step
	if(threadIdx.x == 0) {
		//! \todo do we need to round to block size if multiple chunks per delay?
		s_synapsesPerDelay = ALIGN(s_maxL0SynapsesR, warpSize);
		s_chunksPerDelay = DIV_CEIL(s_synapsesPerDelay, THREADS_PER_BLOCK);
		s_delaysPerChunk = THREADS_PER_BLOCK / s_synapsesPerDelay;
	}
	__syncthreads();

	for(int i=0; i<s_firingCount; ++i) {

		int postsynaptic = s_firingIdx[i];

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_arrivals[MAX_DELAY];

		if(s_recentArrivals[postsynaptic]) {

			//! \todo factor this out and share with integrate step
			if(threadIdx.x == 0) {
				s_delayBlocks = 0;

				/* It's probably not worthwhile pre-loading arrival delays, since
				 * only a few of the loaded values will be used */
				//! \todo could pre-load in one go for all the ones that did fire, though
				uint32_t arrivalBits = g_arrivalDelays[postsynaptic];
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

			for(int chunk=0; chunk < s_chunkCount; ++chunk) {

				int delayEntry = s_delaysPerChunk == 0 ?
					chunk / s_chunksPerDelay :
					chunk * s_delaysPerChunk + threadIdx.x / s_synapsesPerDelay;
				uint32_t delay = s_arrivals[delayEntry];
				/* Offset /within/ a delay block */
				int synapseIdxR = s_delaysPerChunk == 0 ?
					(chunk % s_chunksPerDelay) * THREADS_PER_BLOCK + threadIdx.x :
					(threadIdx.x % s_synapsesPerDelay);

				// reverse matrix *only* contains excitatory neurons
				//! \todo consider using per-neuron maximum here instead
				if(synapseIdxR < s_maxL0SynapsesR 
						&& delayEntry < s_delayBlocks
#ifdef __DEVICE_EMULATION__
						// warp size is 1, so rounding to warp size not as expected
						&& threadIdx.x < s_synapsesPerDelay * s_delaysPerChunk
#endif
					)
				{

					size_t synapseAddressR = 
						postsynaptic * maxDelay * cmPitchR
						+ delay * cmPitchR
						+ synapseIdxR;

					uint sdataR = g_cmR[synapseAddressR];

					if(sdataR != INVALID_REVERSE_SYNAPSE) {

						/* The delivery time of the last spike on this synapse is
						 * recorded in the forward matrix. */
						uint synapseIdxF = forwardIdx(sdataR);

						size_t forwardAddress = 
							sourceNeuron(sdataR) * maxDelay * cmPitchF
							+ delay * cmPitchF
							+ synapseIdxF;

						uint sdataF = g_cmF[forwardAddress];

						int dt = currentTime - arrivalTime(sdataF);
						//assert(dt > 0);
						if(dt < s_stdpTauP) {
							g_ltp[synapseAddressR] += potentiation(dt);
							DEBUG_MSG("ltp +%f for synapse %u -> %u after delay of %u\n",
									potentiation(dt), sourceNeuron(sdataR), postsynaptic, dt);
						}
					}
				}
			}
			__syncthreads();
		}
        __syncthreads();
	}
}




//! \todo factor out and share with deliverL0
__device__
void
setDelayBits(
	// input
	uint32_t delayBits,
	// output
	uint* s_delayBlocks,
	uint32_t* s_delays
)
{
	if(threadIdx.x == 0) {
		uint delayBlocks = 0;
		while(delayBits) {
			int arrivalDelay = __ffs(delayBits) - 1;
			s_delays[delayBlocks] = arrivalDelay;
			delayBits &= ~(0x1 << arrivalDelay);
			delayBlocks += 1;
		}
#if 0
		s_chunkCount = s_delaysPerChunk == 0 ?
			delayBlocks * s_chunksPerDelay :  // >= 1 chunk(s) per delay
			DIV_CEIL(delayBlocks, s_delaysPerChunk);  // multiple delays per chunk
#endif
		*s_delayBlocks = delayBlocks;
	}
	__syncthreads();
}



__device__
void
setPartitionParameters(uint* s_partitionSize, uint* s_neuronsPerThread)
{
    if(threadIdx.x == 0) {
        *s_partitionSize = c_partitionSize[CURRENT_PARTITION];
        *s_neuronsPerThread = DIV_CEIL(*s_partitionSize, THREADS_PER_BLOCK);
	}
	__syncthreads();
}



__global__
void
clearSTDPAccumulator_(
		uint maxPartitionSize,
		uint maxDelay,
		// Delay bits
		uint32_t* g_delayBits,
		size_t pitch32,
		// Accumulator
		float* g_acc,		//! \note caller should point to correct part of multi-dimensional matrix
		size_t pitch,
		size_t size)
{
	//! \todo add timing of this kernel as well

	__shared__ uint s_partitionSize;
	__shared__ uint s_neuronsPerThread;

	setPartitionParameters(&s_partitionSize, &s_neuronsPerThread);

	/* Pre-load all delay bits, since all of it will be needed */
	__shared__ uint32_t s_delayBits[MAX_PARTITION_SIZE];
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);

	//! \todo time this without loading delay bits 
	for(uint presynaptic=0; presynaptic<s_partitionSize; ++presynaptic) {

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];

		setDelayBits(s_delayBits[presynaptic], &s_delayBlocks, s_delays);

		ASSERT(pitch <= THREADS_PER_BLOCK);

		//! \todo deal with several delays in parallel as in L0 delivery (see also addL0LTD)
		//! \todo deal with multiple chunks per delay
		for(uint delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {
			uint delay = s_delays[delayIdx];
			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < pitch) {
				size_t g_offset = (presynaptic * maxDelay + delay) * pitch + threadIdx.x;
				g_acc[g_offset] = 0;
			}
		}
	}
}


/* Re-order long-term potentiation from the reverse order (by postsynaptic)
 * used in the accumulation array, to the forward order (by presynaptic) used
 * in the synaptic weight matrix. 
 *
 * prefix r: reverse matrix
 * prefix f: forward matrix
 */
__global__
void
reorderLTP_(
#ifdef KERNEL_TIMING
	unsigned long long* g_cc,
	size_t ccPitch,
#endif
	int maxPartitionSize,
	int maxDelay,
	size_t pitch32,
	uint32_t* g_delayBits,
	// forward connectivity
	uint* gf_cm,
	size_t f_pitch,
	size_t f_size,
	// reverse connectivity
	uint* gr_cm,
	size_t r_pitch,
	size_t r_size)
{
	SET_COUNTER(s_ccReorderSTDP, 0);

	/* The accumulated long-term potentiation is stored in a reverse-order matrix. */
	__shared__ int s_partitionSize;
	__shared__ int s_neuronsPerThread;
	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		s_neuronsPerThread = DIV_CEIL(s_partitionSize, THREADS_PER_BLOCK);
	}
	__syncthreads();

	/* Pre-load all delay bits, since all of it will be needed */
	__shared__ uint32_t s_delayBits[MAX_PARTITION_SIZE];
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);

	size_t poffset = CURRENT_PARTITION * maxPartitionSize * maxDelay;
	uint* g_raddress =       gr_cm + RCM_ADDRESS * r_size + poffset * r_pitch;
	float* gr_ltp = (float*) gr_cm + RCM_LTP     * r_size + poffset * r_pitch;
	float* gf_ltp = (float*) gf_cm + CM_FLTP     * f_size + poffset * f_pitch;

	for(uint postsynaptic=0; postsynaptic < s_partitionSize; ++postsynaptic) {

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];
		setDelayBits(s_delayBits[postsynaptic], &s_delayBlocks, s_delays);

		ASSERT(r_pitch <= THREADS_PER_BLOCK);

		for(int delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {

			int delay = s_delays[delayIdx];
			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < r_pitch) {
				size_t gr_offset = (postsynaptic * maxDelay + delay) * r_pitch + threadIdx.x;
				uint rsynapse = g_raddress[gr_offset];
				if(rsynapse != INVALID_REVERSE_SYNAPSE) {

					float ltp = gr_ltp[gr_offset];

					if(ltp != 0.0f) {

						size_t gf_offset 
								= (sourceNeuron(rsynapse) * maxDelay + delay) * f_pitch 
								+ forwardIdx(rsynapse);

						gf_ltp[gf_offset] = ltp;
						gr_ltp[gr_offset] = 0;

						DEBUG_MSG("stdp %+f for synapse %u -> %u\n",
							ltp, sourceNeuron(rsynapse), postsynaptic);
					}
				}
			}
		}
	}

	SET_COUNTER(s_ccReorderSTDP, 1);
	WRITE_COUNTERS(s_ccReorderSTDP, g_cc, ccPitch, 2);
}




/*! Apply STDP, i.e. modify synapses using the accumulated LTP and LTD statistics, 
 * modulated by reward. Synapse weights are limited to [0, maxWeight]. Synapses
 * which are already 0, are not potentiated */
__global__
void
applySTDP_(
#ifdef KERNEL_TIMING
	unsigned long long* g_cc,
	size_t ccPitch,
#endif
	float reward,
	float maxWeight,
	int maxPartitionSize, // not warp aligned
	int maxDelay,
	size_t pitch32,
	uint32_t* g_delayBits,
	uint* g_cm,
	size_t pitch,
	size_t size,
	bool recordTrace)
{
	SET_COUNTER(s_ccApplySTDP, 0);

	__shared__ uint s_partitionSize;
	__shared__ uint s_neuronsPerThread;
	setPartitionParameters(&s_partitionSize, &s_neuronsPerThread);

	/* Pre-load all delay bits, since all of it will be needed */
	__shared__ uint32_t s_delayBits[MAX_PARTITION_SIZE];
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);

	size_t partitionOffset = CURRENT_PARTITION * maxPartitionSize * maxDelay * pitch;
#ifdef __DEVICE_EMULATION__
	uint* g_postsynaptic =      g_cm + CM_ADDRESS * size + partitionOffset;
#endif
	float* g_weights = (float*) g_cm + CM_WEIGHT     * size + partitionOffset;
	float* g_ltp     = (float*) g_cm + CM_FLTP       * size + partitionOffset;
	float* g_ltd     = (float*) g_cm + CM_LTD        * size + partitionOffset;
	uint* g_trace    =          g_cm + CM_STDP_TRACE * size + partitionOffset;

	for(uint presynaptic=0; presynaptic<s_partitionSize; ++presynaptic) {

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];

		setDelayBits(s_delayBits[presynaptic], &s_delayBlocks, s_delays);
		ASSERT(pitch <= THREADS_PER_BLOCK);

		//! \todo deal with several delays in parallel as in L0 delivery
		//! \todo deal with multiple chunks per delay
		for(uint delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {
			uint delay = s_delays[delayIdx];

			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < pitch) {

				size_t g_offset 
					= (presynaptic * maxDelay + delay) * pitch 
					+ threadIdx.x;

				float ltp = g_ltp[g_offset];
				float ltd = g_ltd[g_offset];
				float w_diff = reward * (ltp + ltd);

				if(w_diff != 0.0f) {

					float w_old = g_weights[g_offset];
					float w_new = fmin(maxWeight, fmax(w_old + w_diff, 0.0f));

					/* Only modify excitatory synapses. Also, don't modify
					 * weight once it has reached 0. */
					//! \todo for synapses with zero weight, don't write to accumulator in the first place
					if(w_old > 0.0f && w_old != w_new) {
						g_weights[g_offset] = w_new;
						DEBUG_MSG("stdp, updated synapse %u -> %u to %f\n",
							presynaptic, targetNeuron(g_postsynaptic[g_offset]), w_new);

						if(recordTrace) {
							g_trace[g_offset] = __float_as_int(w_new);
						}
					}
				}

				if(ltp != 0.0f) {
					g_ltp[g_offset] = 0.0f;
				}

				if(ltd != 0.0f) {
					g_ltd[g_offset] = 0.0f;
				}

			}
		}
	}
	SET_COUNTER(s_ccApplySTDP, 1);
	WRITE_COUNTERS(s_ccApplySTDP, g_cc, ccPitch, 2);
}
