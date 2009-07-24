#include <vector>
#include <cutil.h>

#include "log.hpp"
#include "error.cu"
//! \todo remove: this is only needed for MASK 
#include "spike.cu"
#include "util.h"


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
	return s_depression[abs(dt)];
}


__device__
float
potentiation(int dt)
{
	return s_potentiation[abs(dt)];
}




/* Update a single synapse according to STDP rule
 *
 * Generally, the synapse is potentiated if the postsynaptic neuron fired
 * shortly after the spike arrived. Conversely, the synapse is depressed if the
 * postsynaptic neuron fired shortly before the spike arrived.
 *
 * Both potentiation and depression is applied with some delay after the neuron
 * actually fired. This is so that all the relevant history has taken place.
 *
 * We determine which synapses are potentiated and which are depressed by
 * inspecting the recent firing history of both the pre and post-synaptic
 * neuron. Consider the firing history at the presynatic neuron:
 *
 *    |---P---||---D---||--delay--|
 * XXXPPPPPPPPPDDDDDDDDDFFFFFFFFFFF
 * 31      23      15      7      0
 *
 * where
 *	X: cycles not of interest as spikes would have reached postsynaptic outside
 *	   the STDP window 
 *  D: (D)epressing spikes which would have reached after postsynaptic firing
 *  P: (P)otentiating spikes which would have reached before postsynaptic firing
 *	F: In-(F)light spikes which have not yet reached.
 *
 * Within the P spikes (if any) we only consider the *last* spike. Within the D
 * spikes (if any) we only consider the *first* spike.
 */
__device__
float
updateSynapse(
		uint r_synapse,
		uint targetNeuron,
		uint rfshift,
		uint32_t* sourceRecentFiring, // L0: shared memory; L1: global memory
		uint32_t* s_targetRecentFiring)
{
	int inFlight
		= rfshift
		+ r_delay(r_synapse) - 1; /* -1 since we do spike arrival before
									 neuron-update and STDP in a single
									 simulation cycle */
	uint32_t sourceFiring 
		= (sourceRecentFiring[sourceNeuron(r_synapse)]
			& ~0x80000000)        // hack to get consistent results (*)
		>> inFlight;

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

	float w_diff = 0.0f;
	uint32_t p_spikes = (sourceFiring >> s_stdpTauD) & MASK(s_stdpTauP);

	if(p_spikes) {
		int dt = __ffs(p_spikes) - 1;

		/* A spike was sent from pre- to post-synaptic. However, it's possible
		 * that this spike has already caused potentiation due to multiple
		 * firings at the postsynaptic */
		bool alreadyPotentiated
			= (s_targetRecentFiring[targetNeuron] >> s_stdpTauD)
			& MASK(dt);

		if(!alreadyPotentiated) {
			w_diff += potentiation(dt);
			DEBUG_MSG("ltp %+f for synapse %u-%u -> %u-%u (dt=%u, delay=%u)\n",
					potentiation(dt),
					sourcePartition(r_synapse), sourceNeuron(r_synapse),
					CURRENT_PARTITION, targetNeuron, dt, r_delay(r_synapse));
		}
	}

	uint32_t d_spikes = sourceFiring & MASK(s_stdpTauD);

	if(d_spikes) {

		int dt = __clz(d_spikes << (32 - rfshift - s_stdpTauD));

		/* A spike was sent from pre- to post-synaptic. However, it's possible
		 * that this spike has already caused depression due to the
		 * postsynaptic firing again between the currently considered firing
		 * and the spike arrival. That depression will be processed in a few
		 * simulation cycles' time. */
		bool alreadyDepressed
			= s_targetRecentFiring[targetNeuron] >> (s_stdpTauD-1-dt)
			& MASK(dt);

		if(!alreadyDepressed) {
			w_diff += depression(dt);
			DEBUG_MSG("ltd: %+f for synapse %u-%u -> %u-%u (dt=%u, delay=%u)\n",
					depression(dt),
					sourcePartition(r_synapse), sourceNeuron(r_synapse),
					CURRENT_PARTITION, targetNeuron,
					dt, r_delay(r_synapse));
		}
	}

	return w_diff;
}



/*! Update STDP statistics for all neurons
 *
 * Process each firing neuron, potentiating synapses with spikes reaching the
 * fired neuron shortly before firing. */
__device__
void
updateSTDP_(
	bool isL1, // hack to work out how to address recent firing bits
	uint32_t* sourceRecentFiring,
	uint32_t* s_targetRecentFiring,
	size_t pitch32,
	uint rfshift, // how much to shift recent firing bits
	uint partitionSize,
	uint r_maxSynapses,
	uint* gr_cm, size_t r_pitch, size_t r_size,
	uint32_t* s_firingIdx) // thread buffer
{
    __shared__ uint s_schunkCount; // number of chunks for synapse-parallel execution
    __shared__ uint s_nchunkCount; // number of chunks for neuron-parallel execution

    //! \todo factor this out and share with integrate step
    if(threadIdx.x == 0) {
        // deal with at most one postsynaptic neuron in one chunk
		s_schunkCount = DIV_CEIL(r_maxSynapses, THREADS_PER_BLOCK); // per-partition size
		s_nchunkCount = DIV_CEIL(partitionSize, THREADS_PER_BLOCK);
    }
    __syncthreads();

    float* gr_stdp = (float*) (gr_cm + RCM_STDP * r_size);
    uint32_t* gr_address = gr_cm + RCM_ADDRESS * r_size;

	/* Determine what postsynaptic neurons needs processing in small batches */
	for(uint nchunk=0; nchunk < s_nchunkCount; ++nchunk) {

		uint target = nchunk * THREADS_PER_BLOCK + threadIdx.x;
		const int processingDelay = s_stdpTauD;
		bool fired = s_targetRecentFiring[target] & (0x1 << processingDelay);

		__shared__ uint s_firingCount;
		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		if(fired && target < partitionSize) {
			uint i = atomicAdd(&s_firingCount, 1);
			s_firingIdx[i] = target;
		}
		__syncthreads();

		for(uint i=0; i<s_firingCount; ++i) {

			uint target = s_firingIdx[i];

			//! \todo consider using per-neuron maximum here instead
			for(uint schunk=0; schunk < s_schunkCount; ++schunk) {

				uint r_sidx = schunk * THREADS_PER_BLOCK + threadIdx.x;

				if(r_sidx < r_maxSynapses) {

					//! \todo move this inside updateSynapse as well
					size_t r_address = target * r_pitch + r_sidx;
					uint r_sdata = gr_address[r_address];

					if(r_sdata != INVALID_REVERSE_SYNAPSE) {

						/* For L0 LTP, recentFiring is in shared memory so access
						 * is cheap. For L1, recentFiring is in a global memory
						 * double buffer. Accesses are both expensive and
						 * non-coalesced. */
						//! \todo consider using a cache for L1 firing history
						float w_diff = updateSynapse(
								r_sdata,
								target,
								rfshift,
								isL1 ? sourceRecentFiring + sourcePartition(r_sdata) * pitch32
								: sourceRecentFiring,
								s_targetRecentFiring);

						//! \todo perhaps stage diff in output buffers
						if(w_diff != 0.0f) {
							gr_stdp[r_address] += w_diff;
						}
					}
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}
