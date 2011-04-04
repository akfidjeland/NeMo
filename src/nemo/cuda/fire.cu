#ifndef NEMO_CUDA_STEP_CU
#define NEMO_CUDA_STEP_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file fire.cu Firing/neuron update kernel */

#include "log.cu_h"
#include "fixedpoint.cu"
#include "bitvector.cu"
#include "localQueue.cu"
#include "thalamicInput.cu"


/*! Set per-neuron bit-vector for fired neurons in both shared and global memory
 *
 * \param[in] nfired
 *		Number of neurons in current partition which fired this cycle.
 * \param[in] s_fired
 *		Vector of indices of the fired neuron. The first \a nfired entries
 *		should be set.
 * \param[out] s_dfired
 *		Per-neuron bit-vector in shared memory for fired neurons.
 * \param[out] g_dfired
 *		Per-neuron bit-vector in global memory for fired neurons.
 *
 * \see loadDenseFiring
 */
__device__
void
storeDenseFiring(unsigned nfired, nidx_dt* s_fired, uint32_t* g_dfired)
{
	__shared__ uint32_t s_dfired[S_BV_PITCH];

	bv_clear_(s_dfired);

	for(unsigned nbase=0; nbase < nfired; nbase += THREADS_PER_BLOCK) {
		unsigned i = nbase + threadIdx.x;
		unsigned neuron = s_fired[i];
		bv_atomicSetPredicated(i < nfired, neuron, s_dfired);
	}
	__syncthreads();

	bv_copy(s_dfired, g_dfired + CURRENT_PARTITION * c_bv_pitch);
}


/*! Store sparse firing in global memory buffer
 *
 * The global memory roundtrip is required to support having 'fire' and
 * 'scatter' in separate kernels.
 *
 * \param[in] nFired number of neurons in this partition which fired this cycle
 * \param[in] s_fired shared memory vector of the relevant neuron indices.
 * \param[out] g_nFired global memory per-partition vector of firing counts
 * \param[out] g_fired global memory per-neuron vector of fired neuron indices.
 * 		For each partition, only the first \a nFired entries contain valid data.
 *
 * \see loadSparseFiring
 */
__device__
void
storeSparseFiring(unsigned nFired, nidx_dt* s_fired, unsigned* g_nFired, nidx_dt* g_fired)
{
	for(unsigned b=0; b < nFired; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < nFired) {
			g_fired[CURRENT_PARTITION * c_pitch32 + i] = s_fired[i];
		}
	}

	if(threadIdx.x == 0) {
		g_nFired[CURRENT_PARTITION] = nFired;
	}
}



/*! The external firing stimulus is (possibly) provided in a per-neuron
 * bit-vector */
__device__
void
loadFiringInput(uint32_t* g_firing, uint32_t* s_firing)
{
	if(g_firing != NULL) {
		bv_copy(g_firing + CURRENT_PARTITION * c_bv_pitch, s_firing);
	} else {
		bv_clear(s_firing);
	}
	__syncthreads();
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
	size_t neuronParametersSize = PARTITION_COUNT * c_pitch32;
	float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	float* g_b = g_neuronParameters + PARAM_B * neuronParametersSize;
	float* g_c = g_neuronParameters + PARAM_C * neuronParametersSize;
	float* g_d = g_neuronParameters + PARAM_D * neuronParametersSize;
	float* g_u = g_neuronState + STATE_U * neuronParametersSize;
	float* g_v = g_neuronState + STATE_V * neuronParametersSize;

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		/* if index space is contigous, no warp divergence here */
		if(bv_isSet(neuron, s_valid)) {

			float v = g_v[neuron];
			float u = g_u[neuron];
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

				DEBUG_MSG_NEURON("c%u %u-%u fired (forced: %u)\n",
						s_cycle, CURRENT_PARTITION, neuron, forceFiring);

				//! \todo consider *only* updating this here, and setting u and v separately
				unsigned i = atomicAdd(s_nFired, 1);

				/* can overwrite current as long as i < neuron. See notes below
				 * on synchronisation and declaration of s_current/s_fired. */
				s_fired[i] = neuron;
			}

			g_v[neuron] = v;
			g_u[neuron] = u;
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
		bool thalamicInputEnabled,
		// neuron state
		float* gf_neuronParameters,
		float* gf_neuronState,
		unsigned* gu_neuronState,
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
#ifdef NEMO_CUDA_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_nFired = 0;
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	bv_clear(s_overflow);
	bv_clear(s_negative);

	__shared__ fix_t s_current[MAX_PARTITION_SIZE];
	copyCurrent(s_partitionSize,
			g_current + CURRENT_PARTITION * c_pitch32,
			s_current);
	__syncthreads();

	addCurrentStimulus(s_partitionSize, c_pitch32, g_istim, s_current, s_overflow, s_negative);
	fx_arrSaturatedToFloat(s_overflow, s_negative, s_current, (float*) s_current);

	/* The random input current might be better computed in a separate kernel,
	 * so that the critical section in the MPI backend (i.e. the fire kernel),
	 * is smaller. */
	if(thalamicInputEnabled) {
		thalamicInput(s_partitionSize, c_pitch32,
				gu_neuronState, gf_neuronParameters, (float*) s_current);
	}
	__syncthreads();

	__shared__ uint32_t s_fstim[S_BV_PITCH];
	loadFiringInput(g_fstim, s_fstim);

	__shared__ uint32_t s_valid[S_BV_PITCH];
	bv_copy(g_valid + CURRENT_PARTITION * c_bv_pitch, s_valid);
	__syncthreads();

	updateNeurons(
			s_partitionSize,
			gf_neuronParameters + CURRENT_PARTITION * c_pitch32,
			gf_neuronState + CURRENT_PARTITION * c_pitch32,
			s_valid,
			(float*) s_current,
			s_fstim,
			&s_nFired,
			s_fired);

	__syncthreads();

	storeDenseFiring(s_nFired, s_fired, g_firingOutput);
	storeSparseFiring(s_nFired, s_fired, g_nFired, g_fired);
}



/*! Wrapper for the __global__ call that performs a single simulation step */
__host__
cudaError_t
update_neurons(
		cudaStream_t stream,
		unsigned partitionCount,
		unsigned cycle,
		bool thalamicInputEnabled,
		float* df_neuronParameters,
		float* df_neuronState,
		unsigned* du_neuronState,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	updateNeurons<<<dimGrid, dimBlock, 0, stream>>>(
			cycle, thalamicInputEnabled,
			df_neuronParameters, df_neuronState, du_neuronState, d_valid,
			d_fstim,   // firing stimulus
			d_istim,   // current stimulus
			d_current, // internal input current
			d_fout, d_nFired, d_fired);

	return cudaGetLastError();
}



#endif
