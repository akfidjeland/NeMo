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
	int currentTime,
	int s_maxL0SynapsesR,
	// reverse connectivity
	uint* g_cmR, size_t cmPitchR, size_t cmSizeR,
	// forward connectivity
	uint* g_cmF, size_t cmPitchF, size_t cmSizeF,
	uint16_t* s_firingIdx,
	int s_firingCount,
	uint32_t* s_recentArrivals,
	uint32_t* g_arrivalDelays
	)
{
	/*! \note This is the maximum number of chunks required for this whole
	 * cluster. It should be possible to reduce this for rows with few
	 * entries. Perhaps better to just save the number of chunks in
	 * constant memory. It would depend on the chunk size, though. */
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
setPartitionParameters(int* s_partitionSize, int* s_neuronsPerThread)
{
    if(threadIdx.x == 0) {
        *s_partitionSize = c_partitionSize[CURRENT_PARTITION];
        *s_neuronsPerThread = DIV_CEIL(*s_partitionSize, THREADS_PER_BLOCK);
	}
	__syncthreads();
}


__global__
void
addL0LTD(
    float reward,
	int maxPartitionSize, // not warp aligned
	int maxDelay,
	// inputs
	size_t pitch32,
	uint32_t* g_delayBits,
	// forward connectivity
	uint* g_cm,
	size_t forwardPitch,
	size_t forwardSize)
{
	/* We assume that in any given modification step most synapses are *not*
	 * updated. We therefore optmisise for read accesses rather than for
	 * writes. With the loops, one forward (LTD) and one reverse (LTP), all
	 * reads are coalesced. Furthermore LTD writes are coalesced. Since we do
	 * two separate updates we need to use global atomics to avoid race
	 * conditions. */

	__shared__ int s_partitionSize;
	__shared__ int s_neuronsPerThread;
	setPartitionParameters(&s_partitionSize, &s_neuronsPerThread);

	/* Pre-load all delay bits, since all of it will be needed */
	__shared__ uint32_t s_delayBits[MAX_PARTITION_SIZE];
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);

	//! \todo factor out common bits here
#ifdef __DEVICE_EMULATION__
	uint* g_postsynaptic = g_cm + CM_ADDRESS * forwardSize
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * forwardPitch;
#endif
	float* g_weights = (float*) g_cm + CM_WEIGHT * forwardSize
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * forwardPitch;
	uint* g_ltd = g_cm + CM_LTD * forwardSize
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * forwardPitch;

	//! \todo experiment with either atomics or manual operations
	for(int presynaptic=0; presynaptic<s_partitionSize; ++presynaptic) {
		// compute the delays for which we have synapses

		//! \todo share this memory with other functions
		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];

		setDelayBits(s_delayBits[presynaptic], &s_delayBlocks, s_delays);
#ifdef __DEVICE_EMULATION__
		//! \todo return error instead here
		assert(forwardPitch <= THREADS_PER_BLOCK);
#endif

		//! \todo deal with several delays in parallel as in L0 delivery (see also constrainWeights)
		//! \todo deal with multiple chunks per delay
		/*! \todo count how many neurons are affected. Perhaps we could use bit
		 * vector to indicate which parts to load. */
		for(int delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {
			int delay = s_delays[delayIdx];
			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < forwardPitch) {
				size_t g_offset = (presynaptic * maxDelay + delay) * forwardPitch + threadIdx.x;
				float ltd = __int_as_float(atomicExch(g_ltd + g_offset, 0));
                //! \todo could do just a write rather than exchange, if reward = 0
				if(ltd != 0.0f && reward != 0.0f) {
					/* Once weight is 0, stay there */
					float oldWeight = g_weights[g_offset];
					if(oldWeight != 0.0f) {
						g_weights[g_offset] = oldWeight + ltd * reward;
#ifdef __DEVICE_EMULATION__
						int postsynaptic = targetNeuron(g_postsynaptic[g_offset]);
						DEBUG_MSG("stdp %+f for synapse %u -> %u\n", ltd, presynaptic, postsynaptic);
#endif
					}
				}
			}
		}
	}
}


__global__
void
addL0LTP(
    float reward,
	int maxPartitionSize,
	int maxDelay,
	size_t pitch32,
	uint32_t* g_delayBits,
	// forward connectivity
	uint* g_cmF,
	size_t cmPitchF,
	size_t cmSizeF,
	// reverse connectivity
	uint* g_cmR,
	size_t cmPitchR,
	size_t cmSizeR)
{
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

	//! \todo factor out common bits here
	uint* g_raddress = g_cmR + RCM_ADDRESS * cmSizeR
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * cmPitchR;
	uint* g_ltp = g_cmR + RCM_LTP * cmSizeR
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * cmPitchR;
	float* g_weights = (float*) g_cmF + CM_WEIGHT * cmSizeF
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * cmPitchF;

	// load reverse delay bits
	for(int postsynaptic=0; postsynaptic < s_partitionSize; ++postsynaptic) {

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];
		setDelayBits(s_delayBits[postsynaptic], &s_delayBlocks, s_delays);
#ifdef __DEVICE_EMULATION__
		//! \todo return error instead here
		assert(cmPitchR <= THREADS_PER_BLOCK);
#endif
		for(int delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {
			int delay = s_delays[delayIdx];
			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < cmPitchR) {
				size_t g_offset = (postsynaptic * maxDelay + delay) * cmPitchR + threadIdx.x;
				uint rsynapse = g_raddress[g_offset];
				if(rsynapse != INVALID_REVERSE_SYNAPSE) {
					float ltp = __int_as_float(atomicExch(g_ltp + g_offset, 0));
                    //! \todo could do just a write rather than exchange, if reward = 0
					if(ltp != 0.0f && reward != 0.0f) {
						uint synapseIdx = forwardIdx(rsynapse);
						size_t weightOffset =
							(sourceNeuron(rsynapse) * maxDelay + delay) * cmPitchF + synapseIdx;
						float weight = g_weights[weightOffset];
						if(weight != 0.0f) { // Once weight is 0, stay there
							g_weights[weightOffset] = weight + ltp * reward;
							DEBUG_MSG("stdp %+f for synapse %u -> %u\n",
								ltp * reward, sourceNeuron(rsynapse), postsynaptic);
						}
					}
				}
			}
		}
	}
}



/* Check every excitatory synapse, making sure it doesn't stray outside limits.
 * This must be done after updates based on LTP and LTD to avoid synapses
 * changing from excitatory to inhibitory, and from going to extremes. */
__global__
void
constrainL0Weights(
	float maxWeight,
	int maxPartitionSize, // not warp aligned
	int maxDelay,
	size_t pitch32,
	uint32_t* g_delayBits,
	uint* g_cm,
	size_t forwardPitch,
	size_t forwardSize)
{
	__shared__ int s_partitionSize;
	__shared__ int s_neuronsPerThread;
	setPartitionParameters(&s_partitionSize, &s_neuronsPerThread);

	/* Pre-load all delay bits, since all of it will be needed */
	__shared__ uint32_t s_delayBits[MAX_PARTITION_SIZE];
	STDP_FN(loadSharedArray)(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);

#ifdef __DEVICE_EMULATION__
	uint* g_postsynaptic = g_cm + CM_ADDRESS * forwardSize
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * forwardPitch;
#endif
	float* g_weights = (float*) g_cm + CM_WEIGHT * forwardSize
					+ CURRENT_PARTITION * maxPartitionSize * maxDelay * forwardPitch;

	for(int presynaptic=0; presynaptic<s_partitionSize; ++presynaptic) {

		__shared__ uint s_delayBlocks;
		__shared__ uint32_t s_delays[MAX_DELAY];

		setDelayBits(s_delayBits[presynaptic], &s_delayBlocks, s_delays);
#ifdef __DEVICE_EMULATION__
		//! \todo return error instead here
		assert(forwardPitch <= THREADS_PER_BLOCK);
#endif

		//! \todo deal with several delays in parallel as in L0 delivery (see also addL0LTD)
		//! \todo deal with multiple chunks per delay
		/*! \todo count how many neurons are affected. Perhaps we could use bit
		 * vector to indicate which parts to load. */
		for(int delayIdx=0; delayIdx<s_delayBlocks; ++delayIdx) {
			int delay = s_delays[delayIdx];
			//! \todo make this work even if there are more threads than delays
			if(threadIdx.x < forwardPitch) {
				size_t g_offset = (presynaptic * maxDelay + delay) * forwardPitch + threadIdx.x;
				float oldWeight = g_weights[g_offset];
				float newWeight = fmin(maxWeight, fmax(oldWeight, 0.0f));
				/* Only modify excitatory synapses */
				if(oldWeight > 0.0f && oldWeight != newWeight) {
					g_weights[g_offset] = newWeight;
					DEBUG_MSG("stdp, limited synapse %u -> %u to %f\n",
						presynaptic, targetNeuron(g_postsynaptic[g_offset]), newWeight);
				}
			}
		}
	}
}
