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

__constant__ float c_depression[MAX_STDP_DELAY];
__constant__ float c_potentiation[MAX_STDP_DELAY];


#define SET_STDP_PARAMETER(symbol, val) CUDA_SAFE_CALL(\
        cudaMemcpyToSymbol(symbol, &val, sizeof(val), 0, cudaMemcpyHostToDevice)\
    )

__host__
void
configureSTDP(int tauP, int tauD,
		std::vector<float>& h_prefire,
		std::vector<float>& h_postfire)
{
    SET_STDP_PARAMETER(c_stdpTauP, tauP);
    SET_STDP_PARAMETER(c_stdpTauD, tauD);

	cudaMemcpyToSymbol(c_potentiation,
			&h_prefire[0],
			sizeof(float)*MAX_STDP_DELAY,
			0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(c_depression,
			&h_postfire[0],
			sizeof(float)*MAX_STDP_DELAY,
			0, cudaMemcpyHostToDevice);
}


/* In the kernel we load the parameters into shared memory. These variables can
 * then be accessed using broadcast */

//! \todo move into vector as other parameters
__shared__ int s_stdpTauP;
__shared__ int s_stdpTauD;


#define LOAD_STDP_PARAMETER(symbol) s_ ## symbol = c_ ## symbol

__shared__ float s_potentiation[MAX_STDP_DELAY];
__shared__ float s_depression[MAX_STDP_DELAY];


__device__
void
loadStdpParameters()
{
    //! \todo could use an array for this and load in parallel
    if(threadIdx.x == 0) {
        LOAD_STDP_PARAMETER(stdpTauP);
        LOAD_STDP_PARAMETER(stdpTauD);
    }
	ASSERT(MAX_STDP_DELAY <= THREADS_PER_BLOCK);
	int dt = threadIdx.x;
	if(dt < MAX_STDP_DELAY) {
		s_potentiation[dt] = c_potentiation[dt];
		s_depression[dt] = c_depression[dt];
	}
    __syncthreads();
}



__device__
float
depression(int dt)
{
	//! \todo ensure dt always of the right sign
	return s_depression[abs(dt)];
}


__device__
float
potentiation(int dt)
{
	return s_potentiation[abs(dt)];
}



/* Depress synapse if the postsynaptic neuron fired shortly before the spike
 * arrived.
 *
 * If two spikes arrive at the postsynaptic neuron after it fired, only the
 * first spike arrival results in depression. To determine if the current spike
 * is the first, consider the firing bits for the presynaptic neuron (one bit
 * per past cycle, LSb is /previous/ cycle, i.e. delay 1):
 *
 *             |--dt---||--delay--|
 * XXXXXXXXXXXXPPPPPPPPPSFFFFFFFFFF
 * 31      23      15      7      0
 *
 * where
 *	X: cycles not of interest as spikes would have reached postsynaptic before
 *	   last firing
 *	P: (P)ast spikes which would have reached after postsynaptic firing and
 *	   before current spike, and thus could cause depression.
 *	S: current (S)pike
 *	F: (F)uture spikes which are in flight but have not yet reached. They will be
 *	   ignored (as far as depression is concerned) when they arrive unless the
 *	   postsynaptic neuron does not fire again.
 *
 * Only 'P' and 'S' cycles are of interest.
 */
__device__
void
depressSynapse(
		uint sourcePartition,         // used for debugging only
		uint sourceNeuron,
		uint targetPartition,         // used for debugging only
		uint targetNeuron,
		uint delay,
		//! \todo just pass in the bits directly? At least for target
		uint32_t* sourceRecentFiring, // L0: shared, L1: global (set to correct partition)
		uint32_t* targetRecentFiring,
		size_t f_offset,              // of synapse, within partition
		float* gf_ltd)                // set to correct partition
{
	//! \todo move s_stdpTauD into dt
	int dt = __ffs(targetRecentFiring[targetNeuron]);

	if(targetRecentFiring[targetNeuron] && abs(dt) < s_stdpTauD) {
		uint32_t p_bits = (~((~0) << dt)) << delay; // see above figure
		uint32_t preSpikes = sourceRecentFiring[sourceNeuron];
		if(!(preSpikes & p_bits)) {

			gf_ltd[f_offset] += depression(dt);

			DEBUG_MSG("ltd: %+f for synapse %u-%u -> %u-%u (dt=%u, delay=%u)\n",
					depression(dt),
					sourcePartition, sourceNeuron,
					targetPartition, targetNeuron,
					dt, delay);
		}
	}
}



/* Potentiate synapse if the postsynaptic neuron fired shortly after a spike
 * arrived.
 *
 * To determine if potentiation should take place we can inspect the firing
 * history of both the presynaptic and postsynaptic neuron.
 *
 * Consider the firing bits for the presynaptic neuron:
 *
 *    |----stdp_max----||--delay--|
 * XXXPPPPPPPPPPPPPPPPPPFFFFFFFFFFF
 *
 * where
 *  I: firings whose spikes are still (I)n flight.
 *  D: firings whose spikes have been (D)elivered and which fall within the STDP
 *     window. Only the most recent of these, if any, is of interest.
 *  X: firings where spike arrival would fall outside STDP window.
 *
 * and for the postsynaptic:
 *                        |--dt--|
 * XXXXXXXXXXXXXXXXXXXXXXXPPPPPPPPF
 *
 * where
 *  F: current postsynaptic firing
 *  P: any previous firings occuring /after/ last spike arrival, which would
 *     have caused a previous potentiation of this synapse.
 *  X: firings before last spike arrival, not of interest.
 *
 * If the postsynaptic neuron fires twice following a spike arrival the
 * relevant synapse(s) are only potentiated based on the delay between the last
 * spike arrival before the postsynaptic firing
 *
 */
__device__
void
potentiateSynapse(
		uint r_synapse,
		uint targetNeuron,
		uint rfshift,
		uint delay,
		uint32_t* sourceRecentFiring, // L0: shared memory; L1: global memory
		uint32_t* s_targetRecentFiring,
		size_t r_offset,
		float* gr_ltp)
{
	//! \todo move s_stdpTau into the mask here.
	// most recent firing which has reached postsynaptic
	int preFired = __ffs((sourceRecentFiring[sourceNeuron(r_synapse)] >> rfshift)
	             & ~0x80000000        // hack to get consistent results (*)
	             & ((~0) << delay));

	/* (*) By the time we deal with LTP we have lost one cycle of history for
	 * the recent firing of the source partition when doing L0. For L1 we read
	 * the recent firing from global memory, so we get 32 cycles worth of
	 * history. For L0 we read from shared memory, which has already been
	 * updated to reflect firing that took place /this/ cycle, so we only get
	 * 31 cycles worth of relevant history. We could, of course, read firing
	 * from global memory in both cases. However, in any event we're short of
	 * history and will loose STDP applications when dt+delay > 32. Untill this
	 * is fixed, the above hack which truncates the history for L1 delivery as
	 * well ensures we get consistent results when modifying the partition
	 * size.  */

	if(preFired) {
		int dt = preFired - delay;
		ASSERT(dt > 0);
		if(dt < s_stdpTauP) {
			/* did this postsynaptic fire in the last dt cycles, i.e. after the
			 * last incoming spike? */
			uint32_t p_mask = ~((~0) << dt) << 1; // see above figure
			bool alreadyPotentiated = s_targetRecentFiring[targetNeuron] & p_mask;
			if(!alreadyPotentiated) {

				gr_ltp[r_offset] += potentiation(dt);

				DEBUG_MSG("ltp %+f for synapse %u-%u -> %u-%u (dt=%u, delay=%u)\n",
						potentiation(dt),
						sourcePartition(r_synapse), sourceNeuron(r_synapse),
						CURRENT_PARTITION, targetNeuron, dt, delay);
			}
		}
	}

}



/*! Process each firing neuron, potentiating synapses with spikes reaching the
 * fired neuron shortly before firing. */
//! \todo fold this into 'fire' loop
__device__
void
updateLTP_(
	bool isL1, // hack to work out how to address recent firing bits
	uint32_t* sourceRecentFiring,
	uint32_t* s_targetRecentFiring,
	size_t pitch32,
	uint rfshift, // how much to shift recent firing bits
	uint r_maxSynapses,
	uint* gr_cm, size_t r_pitch, size_t r_size,
	uint16_t* s_firingIdx,
	uint s_firingCount)
{
    /*! \note This is the maximum number of chunks required for this whole
     * cluster. It should be possible to reduce this for rows with few
     * entries. Perhaps better to just save the number of chunks in
     * constant memory. It would depend on the chunk size, though. */
    __shared__ uint s_chunkCount;

    float* gr_ltp = (float*) (gr_cm + RCM_STDP_LTP * r_size);
    uint32_t* gr_address = gr_cm + RCM_ADDRESS * r_size;

    //! \todo factor this out and share with integrate step
    if(threadIdx.x == 0) {
        // deal with at most one postsynaptic neuron in one chunk
        s_chunkCount = DIV_CEIL(r_maxSynapses, THREADS_PER_BLOCK);
    }
    __syncthreads();

    for(int i=0; i<s_firingCount; ++i) {

        uint target = s_firingIdx[i];

        //! \todo consider using per-neuron maximum here instead
        for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

            uint r_sidx = chunk * THREADS_PER_BLOCK + threadIdx.x;

            if(r_sidx < r_maxSynapses) {

                //! \todo move this inside potentiateSynapse as well
                size_t r_address = target * r_pitch + r_sidx;
                uint r_sdata = gr_address[r_address];

                if(r_sdata != INVALID_REVERSE_SYNAPSE) {

                    /* For L0 LTP, recentFiring is in shared memory so access
                     * is cheap. For L1, recentFiring is in a global memory
                     * double buffer. Accesses are both expensive and
                     * non-coalesced. */
                    //! \todo consider using a cache for L1

                    potentiateSynapse(
                            r_sdata,
                            target,
                            rfshift,
                            r_delay(r_sdata),
                            isL1 ? sourceRecentFiring + sourcePartition(r_sdata) * pitch32
                                 : sourceRecentFiring,
                            s_targetRecentFiring,
                            r_address,
                            gr_ltp);
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
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

    __shared__ uint s_chunkCount;
	__shared__ uint s_partitionSize;

	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
        s_chunkCount = DIV_CEIL(r_pitch, THREADS_PER_BLOCK);
	}
	__syncthreads();

	size_t poffset = CURRENT_PARTITION * maxPartitionSize * r_pitch;
	uint* g_raddress =       gr_cm + RCM_ADDRESS  * r_size + poffset;
	float* gr_ltp = (float*) gr_cm + RCM_STDP_LTP * r_size + poffset;

	float* gf_ltp = (float*) gf_cm + FCM_STDP_LTP * f_size;

	for(uint target=0; target < s_partitionSize; ++target) {
        for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

            uint r_sidx = chunk * THREADS_PER_BLOCK + threadIdx.x;

            if(r_sidx < r_pitch) {

				size_t gr_offset = target * r_pitch + r_sidx;
				uint rsynapse = g_raddress[gr_offset];

				if(rsynapse != INVALID_REVERSE_SYNAPSE) {

					float ltp = gr_ltp[gr_offset];

					if(ltp != 0.0f) {

						//! \todo refactor
						size_t gf_offset
								= sourcePartition(rsynapse) * maxPartitionSize * maxDelay * f_pitch     // partition
								+ (sourceNeuron(rsynapse) * maxDelay + r_delay(rsynapse)-1) * f_pitch   // neuron
								+ forwardIdx(rsynapse);                                                 // synapse

						gf_ltp[gf_offset] = ltp;
						gr_ltp[gr_offset] = 0;

						DEBUG_MSG("stdp %+f for synapse %u-%u -> %u-%u\n", ltp,
							sourcePartition(rsynapse), sourceNeuron(rsynapse),
							CURRENT_PARTITION, target);
					}
				}
			}
		}
        //! \todo remove sync?
		__syncthreads();
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
	loadSharedArray(s_partitionSize, s_neuronsPerThread, pitch32, g_delayBits, s_delayBits);
	__syncthreads();

	size_t partitionOffset = CURRENT_PARTITION * maxPartitionSize * maxDelay * pitch;

#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)
	uint* g_postsynaptic =      g_cm + FCM_ADDRESS    * size + partitionOffset;
#endif
	float* g_weights = (float*) g_cm + FCM_WEIGHT     * size + partitionOffset;
	float* g_ltp     = (float*) g_cm + FCM_STDP_LTP   * size + partitionOffset;
	float* g_ltd     = (float*) g_cm + FCM_STDP_LTD   * size + partitionOffset;
	uint* g_trace    =          g_cm + FCM_STDP_TRACE * size + partitionOffset;

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

					if(w_old != w_new) {

						g_weights[g_offset] = w_new;

						DEBUG_MSG("stdp, updated synapse %u-%u -> %u-%u to %f (%f %f)\n",
							CURRENT_PARTITION, presynaptic,
							targetPartition(g_postsynaptic[g_offset]),
							targetNeuron(g_postsynaptic[g_offset]),
							w_new, ltp, ltd);

						//! \todo conditionally include this
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
		__syncthreads();
	}
	SET_COUNTER(s_ccApplySTDP, 1);
	WRITE_COUNTERS(s_ccApplySTDP, g_cc, ccPitch, 2);
}
