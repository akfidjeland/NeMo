#ifndef NEMO_CUDA_PLUGINS_KURAMOTO_CU
#define NEMO_CUDA_PLUGINS_KURAMOTO_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file Kuramoto.cu Kuramoto oscillator kernel */ 

#include <math.h>

#include <nemo/config.h>
#include <rng.cu_h>
#include <log.cu_h>

#include <bitvector.cu>
#include <neurons.cu>
#include <parameters.cu>
#include <rcm.cu>

#include <nemo/plugins/Kuramoto.h>



//! \todo use more optimal reduction here
__device__
void
sum(float* sdata, float *s_out)
{  
    unsigned tid = threadIdx.x;

    for(unsigned s=THREADS_PER_BLOCK/2; s>0; s>>=1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        } 
        __syncthreads();
    }

    if(tid == 0) {
		*s_out += sdata[0];
	}
}




/*! Compute the influence from other oscillators */
__device__
void
computeIncoming(
	unsigned cycle,
	unsigned partitionSize,
	const param_t& s_params,
	float* g_state,
	rcm_dt g_rcm,
	unsigned target,
	rcm_index_address_t s_row,
	float targetPhase,
	float* s_phaseShift)
{
    unsigned tid = threadIdx.x;
	//! \todo pre-load index addresses for a bunch of targets, and pass this in

	/*! \todo add a second loop and pre-load THREADS_PER_BLOCK warp
	 * addresses */
	for(unsigned bIndex=0 ; bIndex < rcm_indexRowLength(s_row);
			bIndex += THREADS_PER_BLOCK/WARP_SIZE) {

		__shared__ rcm_address_t warp[THREADS_PER_BLOCK/WARP_SIZE];

		/* Incoming phase, all going into a single neuron */
		__shared__ float s_sourcePhase[THREADS_PER_BLOCK];

		if(tid < THREADS_PER_BLOCK/WARP_SIZE) {
			warp[tid] = rcm_address(rcm_indexRowStart(s_row), bIndex + tid, g_rcm);
		}
		__syncthreads();

		size_t r_offset = rcm_offset(warp[tid/WARP_SIZE]);
		rsynapse_t synapse = g_rcm.data[r_offset];
		float incoming = 0.0f;

		if(synapse != INVALID_REVERSE_SYNAPSE) {
			ASSERT(r_delay1(synapse) < MAX_HISTORY_LENGTH-1);
			/* Reading the source state here is non-coalesced. Much of this
			 * should be cachable, however, so for > 2.0 devices we should use
			 * a large L1. */
			float sourcePhase =
				state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(
						int(cycle)-int(r_delay0(synapse)),
						s_params.pitch32,
						sourcePartition(synapse), sourceNeuron(synapse),
						g_state);
			//! \todo check performance difference if using __sinf
			incoming = g_rcm.weights[r_offset] * sinf(sourcePhase-targetPhase);
		}
		s_sourcePhase[tid] = incoming;
		__syncthreads();

		sum(s_sourcePhase, s_phaseShift);
		__syncthreads(); // to protect 'warp'
	}
}




/*! Update state for all oscillators in the current partition */
__global__
void
updateOscillators( 
		uint32_t cycle,
		unsigned* g_partitionSize,
		param_t* g_params,
		float* g_nparams,
		float* g_nstate,
		uint32_t* g_valid,
		rcm_dt g_rcm)
{
	__shared__ unsigned s_partitionSize;
	__shared__ param_t s_params;
	__shared__ uint32_t s_valid[S_BV_PITCH];

	unsigned tid = threadIdx.x;

	loadParameters(g_params, &s_params);
	if(tid == 0) {
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	/* Natural frequency of oscillations */
	const float* g_frequency = g_nparams + CURRENT_PARTITION * s_params.pitch32;

	/* Current phase */
	const float* g_phase0 =
		state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(cycle, s_params.pitch32, g_nstate);

	/* Next phase */
	float* g_phase1 =
		state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(cycle+1, s_params.pitch32, g_nstate);

	/*! \todo consider adding a way to provide external stimulus here */

	for(unsigned bOscillator=0; bOscillator < s_partitionSize; bOscillator += THREADS_PER_BLOCK) {

		__shared__ float s_phase0[THREADS_PER_BLOCK];
		__shared__ float s_phaseShift[THREADS_PER_BLOCK];
		__shared__ rcm_index_address_t s_row[THREADS_PER_BLOCK];

		unsigned oscillator = bOscillator + tid;
		s_phase0[tid] = g_phase0[oscillator];
		s_phaseShift[tid] = 0.0f;
		s_row[tid] = rcm_indexAddress(oscillator, g_rcm);
		__syncthreads();

		/* now compute the incoming phase for each sequentially */
		//! \todo cut loop short when we get to end?
		for(unsigned iTarget=0; iTarget < THREADS_PER_BLOCK; iTarget+= 1) {
			computeIncoming(cycle, s_partitionSize,
					s_params, g_nstate, g_rcm,
					iTarget, s_row[iTarget], s_phase0[iTarget], s_phaseShift + iTarget);
		}
		__syncthreads();

		/* Set next state for THREADS_PER_BLOCK oscillators */
		//! \todo use RK4 here instead
		//! \todo is the validity check really needed here?
		if(bv_isSet(oscillator, s_valid)) {
			float phase = s_phase0[tid]
			            + g_frequency[oscillator] 
			            + s_phaseShift[tid]; 
			DEBUG_MSG_NEURON("phase[%u] (%p+%u) = %f + %f + %f\n",
					oscillator, g_phase1, oscillator,
					s_phase0[tid], g_frequency[oscillator], s_phaseShift[tid]);
			g_phase1[oscillator] = fmodf(phase, 2*M_PI);
		}
	}

	/* Normally the firing is written back to global memory here. The
	 * oscillators do not fire, so just leave it as it is */
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
		nrng_t /* ignored */,
		uint32_t* d_valid,
		uint32_t* /* ignored */,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* /* ignored */,
		unsigned* /* ignored */,
		nidx_dt* /* ignored */,
		rcm_dt* d_rcm)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	updateOscillators<<<dimGrid, dimBlock, 0, stream>>>(
			cycle, d_partitionSize, d_params,
			df_neuronParameters, df_neuronState, d_valid,
			*d_rcm);

	return cudaGetLastError();
}


#endif
