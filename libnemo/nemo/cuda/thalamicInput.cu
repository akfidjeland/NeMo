/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "thalamicInput.cu_h"



__device__
unsigned
neuronLocalStateIndex(unsigned neuron, unsigned plane, size_t pitch)
{
    return (plane * PARTITION_COUNT + CURRENT_PARTITION) * pitch + neuron;
}



//! \todo use unsigned4 instead?
__device__ 
void 
rng_loadState(unsigned *rngState, const unsigned* g_rngState, unsigned neuron, size_t pitch)
{
	for(unsigned i=0; i<4; i++){
		rngState[i] = g_rngState[neuronLocalStateIndex(neuron, i, pitch)];
	}
}


__device__ 
void 
rng_saveState(const unsigned *rngState, unsigned *g_rngState, unsigned neuron, size_t pitch)
{
	for(unsigned i=0; i<4; i++){
		g_rngState[neuronLocalStateIndex(neuron, i, pitch)] = rngState[i];
	}
}


__device__ 
unsigned 
rng_genUniform(unsigned *rngState)
{
	unsigned t = (rngState[0]^(rngState[0]<<11));
	rngState[0] = rngState[1];
	rngState[1] = rngState[2];
	rngState[2] = rngState[3];
	rngState[3] = (rngState[3]^(rngState[3]>>19))^(t^(t>>8));
	return rngState[3];
}



/* For various reasons this generates a pair of samples for each call. If nesc.
 * then you can just stash one of them somewhere until the next time it is
 * needed or something.  */
__device__
float2
rng_genGaussian(unsigned* rngState)
{
	float a = rng_genUniform(rngState) * 1.4629180792671596810513378043098e-9f;
	float b = rng_genUniform(rngState) * 0.00000000023283064365386962890625f;
	float r = sqrtf(-2*logf(1-b));
	return make_float2(sinf(a)*r, cosf(a)*r);
}


__device__
void
thalamicInput(
		size_t partitionSize,
		size_t pitch,
		unsigned* g_rngState,
		float* g_nparam,
		float* s_current)
{
	unsigned rngState[4];

	float* g_sigma = g_nparam +
			+ PARAM_SIGMA * PARTITION_COUNT * pitch
			+ CURRENT_PARTITION * pitch;

	for(unsigned nbase=0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		/* Copy the input state from memory into our local state */
		rng_loadState(rngState, g_rngState, neuron, pitch);


		if(neuron < partitionSize) {

			//! \todo make use of  both randoms
			float2 r = rng_genGaussian(rngState);

			/*! \bug It seems that if r.x is very small the result of the
			 * multiplication is NaN (at least if sigma is 0, possibly in other
			 * cases as well). This issue seems unrelated to fusing
			 * multiply-add operations. Forcing separate multiplication does
			 * not fix. For instance with sigma=0 and r.x=0x00007f80
			 * (4.57384e-41) we get a NaN result. Could not reproduce in
			 * standalone test-case, however. */
			//! \todo consider using fixed-point arithmetic here as well.
			s_current[neuron] += r.x * g_sigma[neuron];
		}

		/* Copy the current RNG state back to memory (not strictly necessary, you
		 * can just generate a new random state every time if you want). */
		rng_saveState(rngState, g_rngState, neuron, pitch);

	}

	__syncthreads();
}
