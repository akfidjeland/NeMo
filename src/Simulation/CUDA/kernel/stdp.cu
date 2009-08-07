#include <vector>
#include <cutil.h>

#include "log.hpp"
#include "error.cu"
#include "util.h"


/* STDP parameters
 *
 * The STDP parameters apply to all neurons in a network. One might, however,
 * want to do this differently for different neuron populations. This is not
 * yet supported.
 */


/* The STDP window is separated into two distinct region: potentiation and
 * depression. In the common assymetrical case the pre-fire part of the STDP
 * window will be potentiation, while the post-fire part of the window will be
 * depression. Other schemes are supported, however. */

/*! The STDP parameters are stored in constant memory, and is loaded into shared
 * memory during execution. \a configureStdp should be called before simulation
 * starts, and \a loadStdpParameters should be called at the beginning of the
 * kernel */


/* Mask of cycles within the STDP for which we consider potentiating synapses */
__constant__ uint64_t c_stdpPotentiation;

/* Mask of cycles within the STDP for which we consider depressing synapses */
__constant__ uint64_t c_stdpDepression;

/* The STDP function sampled at integer cycle points within the STDP window */
__constant__ float c_stdpFn[STDP_WINDOW_SIZE];

/* Length of the window (in cycles) which is post firing */
__constant__ uint c_stdpPostFireWindow;

/* Length of the window (in cycles) which is pre firing */
__constant__ uint c_stdpPreFireWindow;

__constant__ uint c_stdpWindow;


__shared__ uint64_t s_stdpPotentiation;
__shared__ uint64_t s_stdpDepression;
__shared__ float s_stdpFn[STDP_WINDOW_SIZE];
__shared__ uint s_stdpPostFireWindow;
__shared__ uint s_stdpPreFireWindow;


#define SET_STDP_PARAMETER(symbol, val) CUDA_SAFE_CALL(\
        cudaMemcpyToSymbol(symbol, &val, sizeof(val), 0, cudaMemcpyHostToDevice)\
    )


__host__
void
configureStdp(
		uint preFireWindow,
		uint postFireWindow,
		uint64_t potentiationBits, // remainder are depression
		uint64_t depressionBits, // remainder are depression
		float* stdpFn)
{
	SET_STDP_PARAMETER(c_stdpPreFireWindow, preFireWindow);
	SET_STDP_PARAMETER(c_stdpPostFireWindow, postFireWindow);
	SET_STDP_PARAMETER(c_stdpWindow, preFireWindow + postFireWindow);
	SET_STDP_PARAMETER(c_stdpPotentiation, potentiationBits);
	SET_STDP_PARAMETER(c_stdpDepression, depressionBits);
	uint window = preFireWindow + postFireWindow;
	assert(window <= STDP_WINDOW_SIZE);
	cudaMemcpyToSymbol(c_stdpFn, stdpFn,
			sizeof(float)*window,
			0, cudaMemcpyHostToDevice);
}



/* In the kernel we load the parameters into shared memory. These variables can
 * then be accessed using broadcast */



#define LOAD_STDP_PARAMETER(symbol) s_ ## symbol = c_ ## symbol


__device__
void
loadStdpParameters()
{
    if(threadIdx.x == 0) {
        LOAD_STDP_PARAMETER(stdpPotentiation);
        LOAD_STDP_PARAMETER(stdpDepression);
        LOAD_STDP_PARAMETER(stdpPreFireWindow);
        LOAD_STDP_PARAMETER(stdpPostFireWindow);
    }

	ASSERT(MAX_STDP_DELAY <= THREADS_PER_BLOCK);
	int dt = threadIdx.x;
	if(dt < STDP_WINDOW_SIZE) {
		s_stdpFn[dt] = c_stdpFn[dt];
	}
    __syncthreads();
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


#define STDP_NO_APPLICATION (~0)


/*! \return
 *		shortest delay between spike arrival and firing of this neuron or
 *		largest representable delay if STDP not appliccable.
 *
 * STDP is not applicable if the postsynaptic neuron also fired closer to the
 * incoming spike than the firing currently under consideration. */
__device__
uint
closestPreFire(uint64_t spikes, uint targetNeuron)
{
	int dt =  __ffsll(spikes >> s_stdpPostFireWindow);
	return dt ? (uint) dt-1 : STDP_NO_APPLICATION;
}



__device__
uint
closestPostFire(uint64_t spikes, uint rfshift, uint targetNeuron)
{
	int dt = __clzll(spikes << (64 - rfshift - s_stdpPostFireWindow));
	return dt ? (uint) dt : STDP_NO_APPLICATION;
}



#ifdef __DEVICE_EMULATION__

__device__
void
logStdp(int dt, float w_diff, uint targetNeuron, uint32_t r_synapse)
{
	const char* type[] = { "ltd", "ltp" };

	if(w_diff != 0.0f) {
		fprintf(stderr, "%s %+f for synapse %u-%u -> %u-%u (dt=%d, delay=%u)\n",
				type[w_diff > 0.0f], w_diff,
				sourcePartition(r_synapse), sourceNeuron(r_synapse),
				CURRENT_PARTITION, targetNeuron, dt, r_delay(r_synapse));
	}
}

#endif


__device__
float
updateRegion(uint64_t spikes,
		uint rfshift,
		uint targetNeuron,
		uint32_t r_synapse, // used for logging only
		uint64_t* s_targetRecentFiring)
{
	/* The potentiation can happen on either side of the firing. We want to
	 * find the one closest to the firing. We therefore need to compute the
	 * prefire and postfire dt's separately. */
	uint dt_pre = closestPreFire(spikes, targetNeuron);
	uint dt_post = closestPostFire(spikes, rfshift, targetNeuron);

	/* For logging. Positive values: post-fire, negative values: pre-fire */
	int dt_log;

	float w_diff = 0.0f;
	if(spikes) {
		if(dt_pre < dt_post) {
			w_diff = s_stdpFn[s_stdpPreFireWindow - 1 - dt_pre];
			dt_log = -dt_pre;
		} else if(dt_post < dt_pre) {
			w_diff = s_stdpFn[s_stdpPreFireWindow+dt_post];
			dt_log = dt_post;
		}
		// if neither is applicable dt_post == dt_pre
	}
#ifdef __DEVICE_EMULATION__
	logStdp(dt_log, w_diff, targetNeuron, r_synapse);
#endif
	return w_diff;
}



/*! Update a synapse according to the user-specified STDP function. Both
 * potentiation and depression takes place.
 *
 * \return weight modifcation (additive term)
 */
__device__
float
updateSynapse(
		uint32_t r_synapse,
		uint targetNeuron,
		uint rfshift,
		uint64_t* x_sourceFiring, // L0: shared memory; L1: global memory
		uint64_t* s_targetFiring)
{
	int inFlight = rfshift + r_delay(r_synapse) - 1;
	/* -1 since we do spike arrival before neuron-update and STDP in a single
	 * simulation cycle */

	uint64_t sourceFiring = x_sourceFiring[sourceNeuron(r_synapse)] >> inFlight;
	uint64_t p_spikes = sourceFiring & s_stdpPotentiation;
	uint64_t d_spikes = sourceFiring & s_stdpDepression;

	return updateRegion(p_spikes, rfshift, targetNeuron, r_synapse, s_targetFiring)
           + updateRegion(d_spikes, rfshift, targetNeuron, r_synapse, s_targetFiring);
}



/*! Update STDP statistics for all neurons */
__device__
void
updateSTDP_(
	bool isL1, // hack to work out how to address recent firing bits
	uint64_t* sourceRecentFiring,
	uint64_t* s_targetRecentFiring,
	size_t pitch64,
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
		const int processingDelay = s_stdpPostFireWindow;

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
					uint32_t r_sdata = gr_address[r_address];

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
								isL1 ? sourceRecentFiring + sourcePartition(r_sdata) * pitch64
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
