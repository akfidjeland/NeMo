/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file rng.cu Per-neuron random number generation */

#include "kernel.cu_h"
#include "rng.cu_h"


__device__
unsigned
neuronLocalStateIndex(unsigned neuron, unsigned plane, size_t pitch)
{
    return (plane * PARTITION_COUNT + CURRENT_PARTITION) * pitch + neuron;
}



__device__ 
void 
rng_loadState(unsigned *rngState, const unsigned* g_nstate, unsigned neuron, size_t pitch)
{
	for(unsigned i=0; i < 4; i++){
		rngState[i] = g_nstate[neuronLocalStateIndex(neuron, i, pitch)];
	}
}


__device__ 
void 
rng_saveState(const unsigned *rngState, unsigned *g_nstate, unsigned neuron, size_t pitch)
{
	for(unsigned i=0; i < 4; i++){
		g_nstate[neuronLocalStateIndex(neuron, i, pitch)] = rngState[i];
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



/*! Generate a gaussian random number for a specific neuron */
__device__
float
rng_nrand(unsigned neuron, nrng_t g_nrng)
{
	unsigned rngState[4];

	/* Copy the input state from memory into our local state */
	rng_loadState(rngState, g_nrng.state, neuron, g_nrng.pitch);

	float2 r = rng_genGaussian(rngState);

	/* Copy the current RNG state back to memory (not strictly necessary, you
	 * can just generate a new random state every time if you want). */
	rng_saveState(rngState, g_nrng.state, neuron, g_nrng.pitch);

	/* We're throwing away one random number here */
	return r.x;
}
