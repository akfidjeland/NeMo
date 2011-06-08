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


/*! \return offset into the RNG state vector for a given \a neuron and \a plane */
__device__
unsigned*
rng_state(unsigned neuron, unsigned plane, nrng_t g_nrng)
{
    return g_nrng.state + (plane * PARTITION_COUNT + CURRENT_PARTITION) * g_nrng.pitch + neuron;
}



__device__ 
unsigned 
rng_urand(unsigned state[])
{
	unsigned t = (state[0]^(state[0]<<11));
	state[0] = state[1];
	state[1] = state[2];
	state[2] = state[3];
	state[3] = (state[3]^(state[3]>>19))^(t^(t>>8));
	return state[3];
}



/* For various reasons this generates a pair of samples for each call. If nesc.
 * then you can just stash one of them somewhere until the next time it is
 * needed or something.  */
__device__
float2
rng_nrand(unsigned state[])
{
	float a = rng_urand(state) * 1.4629180792671596810513378043098e-9f;
	float b = rng_urand(state) * 0.00000000023283064365386962890625f;
	float r = sqrtf(-2*logf(1-b));
	return make_float2(sinf(a)*r, cosf(a)*r);
}



/*! Generate a gaussian random number from N(0,1) for a specific neuron */
__device__
float
nrand(unsigned neuron, nrng_t g_nrng)
{
	unsigned state[4];

	/* Copy the input state from memory into our local state */
	for(unsigned i=0; i < 4; i++){
		state[i] = *rng_state(neuron, i, g_nrng);
	}

	float2 r = rng_nrand(state);

	/* Copy the current RNG state back to memory (not strictly necessary, you
	 * can just generate a new random state every time if you want). */
	for(unsigned i=0; i < 4; i++){
		*rng_state(neuron, i, g_nrng) = state[i];
	}

	/* We're throwing away one random number here */
	return r.x;
}
