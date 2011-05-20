#ifndef NEMO_CUDA_PLUGINS_IZHIKEVICH_CU
#define NEMO_CUDA_PLUGINS_IZHIKEVICH_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file Izhikevich.cu Izhikevich neuron update kernel */

#include <nemo/config.h>
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include <log.cu_h>
#endif
#include <bitvector.cu>
#include <current.cu>
#include <firing.cu>
#include <fixedpoint.cu>
#include <neurons.cu>
#include <parameters.cu>
#include <rng.cu>

#include <nemo/plugins/Izhikevich.h>


__device__
void
thalamicInput(
		size_t partitionSize,
		size_t pitch,
		float* g_nparam,    // not offset
		nrng_t g_nrng,
		float* s_current)   // correctly offset for this partition
{
	//! \todo make the partition offset in call (for consistency).
	//! \todo make this call via a generic nvector function
	float* g_sigma = g_nparam
			+ PARAM_SIGMA * PARTITION_COUNT * pitch
			+ CURRENT_PARTITION * pitch;

	for(unsigned nbase=0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {
		unsigned neuron = nbase + threadIdx.x;
		float sigma = g_sigma[neuron];
		if(neuron < partitionSize && sigma != 0.0f) {
			s_current[neuron] += nrand(neuron, g_nrng) * sigma;
		}
	}
}



/*! Update state of all neurons
 *
 * Update the state of all neurons in partition according to the equations in
 * Izhikevich's 2003 paper based on
 *
 * - the neuron parameters (a-d)
 * - the neuron state (u, v)
 * - input current (from other neurons, random input current, or externally provided)
 * - per-neuron specific firing stimulus
 *
 * The neuron state is updated using the Euler method.
 *
 * \param[in] s_partitionSize
 *		number of neurons in current partition
 * \param[in] g_neuronParameters
 *		global memory containing neuron parameters (see \ref nemo::cuda::Neurons)
 * \param[in] g_neuronState
 *		global memory containing neuron state (see \ref nemo::cuda::Neurons)
 * \param[in] s_current
 *		shared memory vector containing input current for all neurons in
 *		partition
 * \param[in] s_fstim
 *		shared memory bit vector where set bits indicate neurons which should
 *		be forced to fire
 * \param[out] s_nFired
 *		output variable which will be set to the number of	neurons which fired
 *		this cycle
 * \param[out] s_fired
 *		shared memory vector containing local indices of neurons which fired.
 *		s_fired[0:s_nFired-1] will contain valid data, whereas remaining
 *		entries may contain garbage.
 */
__device__
void
updateNeurons(
	uint32_t cycle,
	const param_t& s_params,
	unsigned s_partitionSize,
	float* g_neuronParameters,
	float* g_neuronState,
	uint32_t* s_valid,   // bitvector for valid neurons
	// input
	float* s_current,    // input current
	// buffers
	uint32_t* s_fstim,
	// output
	unsigned* s_nFired,
	nidx_dt* s_fired)    // s_NIdx, so can handle /all/ neurons firing
{
	//! \todo could set these in shared memory
	size_t neuronParametersSize = PARTITION_COUNT * s_params.pitch32;
	const float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	const float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	const float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	const float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	//! \todo avoid repeated computation of the same data here
	const float* g_u0 = state<1, 2, STATE_U>(cycle, s_params.pitch32, g_neuronState);
	const float* g_v0 = state<1, 2, STATE_V>(cycle, s_params.pitch32, g_neuronState);
	float* g_u1 = state<1, 2, STATE_U>(cycle+1, s_params.pitch32, g_neuronState);
	float* g_v1 = state<1, 2, STATE_V>(cycle+1, s_params.pitch32, g_neuronState);

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		/* if index space is contigous, no warp divergence here */
		if(bv_isSet(neuron, s_valid)) {

			float v = g_v0[neuron];
			float u = g_u0[neuron];
			float a = g_a[neuron];
			float b = g_b[neuron];
			float I = s_current[neuron];

			/* n sub-steps for numerical stability, with u held */
			bool fired = false;
			for(int j=0; j < 4; ++j) {
				if(!fired) { 
					v += 0.25f * ((0.04f*v + 5.0f) * v + 140.0f - u + I);
					/*! \todo: could pre-multiply this with a, when initialising memory */
					u += 0.25f * (a * ( b*v - u ));
					fired = v >= 30.0f;
				} 
			}

			bool forceFiring = bv_isSet(neuron, s_fstim); // (smem broadcast)

			if(fired || forceFiring) {

				/* Only a subset of the neurons fire and thus require c/d
				 * fetched from global memory. One could therefore deal with
				 * all the fired neurons separately. This was found, however,
				 * to slow down the fire step by 50%, due to extra required
				 * synchronisation.  */
				//! \todo could probably hard-code c
				v = g_c[neuron];
				u += g_d[neuron];

#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
				DEBUG_MSG_NEURON("c%u %u-%u fired (forced: %u)\n",
						s_cycle, CURRENT_PARTITION, neuron, forceFiring);
#endif

				//! \todo consider *only* updating this here, and setting u and v separately
				unsigned i = atomicAdd(s_nFired, 1);

				/* can overwrite current as long as i < neuron. See notes below
				 * on synchronisation and declaration of s_current/s_fired. */
				s_fired[i] = neuron;
			}

			g_v1[neuron] = v;
			g_u1[neuron] = u;
		}

		/* synchronise to ensure accesses to s_fired and s_current (which use
		 * the same underlying buffer) do not overlap. Even in the worst case
		 * (all neurons firing) the write to s_fired will be at least one
		 * before the first unconsumed s_current entry. */
		__syncthreads();
	}
}



/*! \brief Perform a single simulation step
 *
 * A simulation step consists of five main parts:
 *
 * - gather incoming current from presynaptic firing for each neuron (\ref gather)
 * - add externally or internally provided input current for each neuron
 * - update the neuron state (\ref fire)
 * - enque outgoing spikes for neurons which fired (\ref scatterLocal and \ref scatterGlobal)
 * - accumulate STDP statistics
 *
 * The data structures involved in each of these stages are documentated more
 * with the individual functions and in \ref cuda_delivery.
 */
__global__
void
updateNeurons(
		uint32_t cycle,
		unsigned* g_partitionSize,
		param_t* g_params,
		// neuron state
		float* gf_neuronParameters,
		float* gf_neuronState,
		nrng_t g_nrng,
		uint32_t* g_valid,
		// firing stimulus
		uint32_t* g_fstim,
		fix_t* g_istim,
		fix_t* g_current,
		uint32_t* g_firingOutput, // dense output, already offset to current cycle
		unsigned* g_nFired,       // device-only buffer
		nidx_dt* g_fired)         // device-only buffer, sparse output
{
	__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];

	/* Per-neuron bit-vectors. See bitvector.cu for accessors */
	__shared__ uint32_t s_overflow[S_BV_PITCH];
	__shared__ uint32_t s_negative[S_BV_PITCH];

	__shared__ unsigned s_nFired;
	__shared__ unsigned s_partitionSize;

	if(threadIdx.x == 0) {
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_nFired = 0;
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	__shared__ param_t s_params;
	loadParameters(g_params, &s_params);

	bv_clear(s_overflow);
	bv_clear(s_negative);

	__shared__ fix_t s_current[MAX_PARTITION_SIZE];
	copyCurrent(s_partitionSize,
			g_current + CURRENT_PARTITION * s_params.pitch32,
			s_current);
	__syncthreads();

	addCurrentStimulus(s_partitionSize, s_params.pitch32, g_istim, s_current, s_overflow, s_negative);
	fx_arrSaturatedToFloat(s_overflow, s_negative, s_current, (float*) s_current, s_params.fixedPointScale);

	/* The random input current might be better computed in a separate kernel,
	 * so that the critical section in the MPI backend (i.e. the neuron update
	 * kernel), is smaller. */
	thalamicInput(s_partitionSize, s_params.pitch32,
			gf_neuronParameters, g_nrng, (float*) s_current);
	__syncthreads();

	__shared__ uint32_t s_fstim[S_BV_PITCH];
	loadFiringInput(s_params.pitch1, g_fstim, s_fstim);

	__shared__ uint32_t s_valid[S_BV_PITCH];
	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	updateNeurons(
			cycle,
			s_params,
			s_partitionSize,
			//! \todo use consistent parameter passing scheme here
			gf_neuronParameters + CURRENT_PARTITION * s_params.pitch32,
			gf_neuronState,
			s_valid,
			(float*) s_current,
			s_fstim,
			&s_nFired,
			s_fired);

	__syncthreads();

	storeDenseFiring(s_nFired, s_params.pitch1, s_fired, g_firingOutput);
	storeSparseFiring(s_nFired, s_params.pitch32, s_fired, g_nFired, g_fired);
}



/*! Wrapper for the __global__ call that performs a single simulation step */
extern "C"
NEMO_PLUGIN_DLL_PUBLIC
cudaError_t
cuda_update_neurons(
		cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* df_neuronParameters,
		float* df_neuronState,
		nrng_t d_nrng,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		struct rcm_dt* /* unused */)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	updateNeurons<<<dimGrid, dimBlock, 0, stream>>>(
			cycle, d_partitionSize, d_params,
			df_neuronParameters, df_neuronState, d_nrng, d_valid,
			d_fstim,   // firing stimulus
			d_istim,   // current stimulus
			d_current, // internal input current
			d_fout, d_nFired, d_fired);

	return cudaGetLastError();
}


#endif
