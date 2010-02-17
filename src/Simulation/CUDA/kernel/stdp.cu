#include "log.hpp"
#include "error.cu"
#include "cycle.cu"
#include "util.h"
#include "fixedpoint.hpp"


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
__constant__ weight_dt c_stdpFn[STDP_WINDOW_SIZE];

/* Length of the window (in cycles) which is post firing */
__constant__ uint c_stdpPostFireWindow;

/* Length of the window (in cycles) which is pre firing */
__constant__ uint c_stdpPreFireWindow;

__constant__ uint c_stdpWindow;


__shared__ uint64_t s_stdpPotentiation;
__shared__ uint64_t s_stdpDepression;
__shared__ weight_dt s_stdpFn[STDP_WINDOW_SIZE];
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
		weight_dt* stdpFn
		)
{
	SET_STDP_PARAMETER(c_stdpPreFireWindow, preFireWindow);
	SET_STDP_PARAMETER(c_stdpPostFireWindow, postFireWindow);
	SET_STDP_PARAMETER(c_stdpWindow, preFireWindow + postFireWindow);
	SET_STDP_PARAMETER(c_stdpPotentiation, potentiationBits);
	SET_STDP_PARAMETER(c_stdpDepression, depressionBits);
	uint window = preFireWindow + postFireWindow;
	assert(window <= STDP_WINDOW_SIZE);
	cudaMemcpyToSymbol(c_stdpFn, stdpFn, sizeof(weight_dt)*window, 0, cudaMemcpyHostToDevice);
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
closestPreFire(uint64_t spikes)
{
	int dt =  __ffsll(spikes >> s_stdpPostFireWindow);
	return dt ? (uint) dt-1 : STDP_NO_APPLICATION;
}



__device__
uint
closestPostFire(uint64_t spikes)
{
	int dt = __clzll(spikes << (64 - s_stdpPostFireWindow));
	return spikes ? (uint) dt : STDP_NO_APPLICATION;
}



#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)

__device__
void
logStdp(int dt, float w_diff, uint targetNeuron, uint32_t r_synapse)
{
	const char* type[] = { "ltd", "ltp" };

	if(w_diff != 0.0f) {
		fprintf(stderr, "c%u %s: %u-%u -> %u-%u %+f (dt=%d, delay=%u, prefire@%u, postfire@%u)\n",
				s_cycle, type[w_diff > 0.0f],
				sourcePartition(r_synapse), sourceNeuron(r_synapse),
				CURRENT_PARTITION, targetNeuron, dt, r_delay1(r_synapse),
				w_diff,
				s_cycle - s_stdpPostFireWindow + dt,
				s_cycle - s_stdpPostFireWindow);
	}
}

#endif


__device__
weight_dt
updateRegion(
		uint64_t spikes,
		uint targetNeuron,
		uint32_t r_synapse) // used for logging only
{
	/* The potentiation can happen on either side of the firing. We want to
	 * find the one closest to the firing. We therefore need to compute the
	 * prefire and postfire dt's separately. */
	uint dt_pre = closestPreFire(spikes);
	uint dt_post = closestPostFire(spikes);

	/* For logging. Positive values: post-fire, negative values: pre-fire */
#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)
	int dt_log;
#endif

	weight_dt w_diff = 0;
	if(spikes) {
		if(dt_pre < dt_post) {
			w_diff = s_stdpFn[s_stdpPreFireWindow - 1 - dt_pre];
#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)
			dt_log = -dt_pre;
#endif
		} else if(dt_post < dt_pre) {
			w_diff = s_stdpFn[s_stdpPreFireWindow+dt_post];
#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)
			dt_log = dt_post;
#endif
		}
		// if neither is applicable dt_post == dt_pre
	}
#if defined(__DEVICE_EMULATION__) && defined(VERBOSE)
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
weight_dt
updateSynapse(
		uint32_t r_synapse,
		uint targetNeuron,
		uint64_t* g_sourceFiring)
{
	int inFlight = r_delay0(r_synapse);
	/* -1 since we do spike arrival before neuron-update and STDP in a single
	 * simulation cycle */

	uint64_t sourceFiring = g_sourceFiring[sourceNeuron(r_synapse)] >> inFlight;
	uint64_t p_spikes = sourceFiring & s_stdpPotentiation;
	uint64_t d_spikes = sourceFiring & s_stdpDepression;

	return updateRegion(p_spikes, targetNeuron, r_synapse)
           + updateRegion(d_spikes, targetNeuron, r_synapse);
}



/*! Update STDP statistics for all neurons */
__device__
void
updateSTDP_(
	uint32_t cycle,
	uint32_t* s_dfired,
	uint64_t* g_recentFiring,
	size_t pitch64,
	uint partitionSize,
	DEVICE_UINT_PTR_T* cr_address,
	DEVICE_UINT_PTR_T* cr_stdp,
	DEVICE_UINT_PTR_T* cr_pitch,
	dnidx_t* s_firingIdx) // s_NIdx, so can handle /all/ neurons firing
{
	__shared__ uint s_schunkCount; // number of chunks for synapse-parallel execution
	__shared__ uint s_nchunkCount; // number of chunks for neuron-parallel execution

	uint r_maxSynapses = cr_pitch[CURRENT_PARTITION];

	if(threadIdx.x == 0) {
		// deal with at most one postsynaptic neuron in one chunk
		s_schunkCount = DIV_CEIL(r_maxSynapses, THREADS_PER_BLOCK); // per-partition size
		//! \todo simplify logic here. No need for division
		s_nchunkCount = DIV_CEIL(partitionSize, THREADS_PER_BLOCK);
	}
	__syncthreads();

	/* Determine what postsynaptic neurons needs processing in small batches */
	for(uint nchunk=0; nchunk < s_nchunkCount; ++nchunk) {

		uint target = nchunk * THREADS_PER_BLOCK + threadIdx.x;

		uint64_t targetRecentFiring =
			g_recentFiring[(readBuffer(cycle) * PARTITION_COUNT + CURRENT_PARTITION) * s_pitch64 + target];

		const int processingDelay = s_stdpPostFireWindow - 1;

		bool fired = targetRecentFiring & (0x1 << processingDelay);

		/* Write updated history to double buffer */
		g_recentFiring[(writeBuffer(cycle) * PARTITION_COUNT + CURRENT_PARTITION) * s_pitch64 + target] =
				(targetRecentFiring << 1) | (bv_isSet(target, s_dfired) ? 0x1 : 0x0);

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

					size_t r_offset = target * r_maxSynapses + r_sidx;
					/* nvcc will warn that gr_address defaults to gmem, as it
					 * is not clear what address space it belongs to. That's
					 * ok; this is global memory */
					uint32_t* gr_address = (uint32_t*) cr_address[CURRENT_PARTITION];
					uint32_t r_sdata = gr_address[r_offset];

					if(r_sdata != INVALID_REVERSE_SYNAPSE) {

						/* For L0 LTP, recentFiring is in shared memory so access
						 * is cheap. For L1, recentFiring is in a global memory
						 * double buffer. Accesses are both expensive and
						 * non-coalesced. */
						//! \todo consider using a cache for L1 firing history
						weight_dt w_diff =
							updateSynapse(
								r_sdata,
								target,
								g_recentFiring + (readBuffer(cycle) * PARTITION_COUNT + sourcePartition(r_sdata)) * pitch64);

						//! \todo perhaps stage diff in output buffers
						//! \todo add saturating arithmetic here
						if(w_diff != 0) {
							((weight_dt*) cr_stdp[CURRENT_PARTITION])[r_offset] += w_diff;
						}
					}
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}
