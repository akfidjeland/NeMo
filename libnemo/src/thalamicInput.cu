#include "kernel.cu_h"
#include "thalamicInput.cu_h"

__device__ 
unsigned 
threadLocalStateIndex(unsigned plane, size_t pitch)
{
    return (plane * PARTITION_COUNT + CURRENT_PARTITION) * pitch + threadIdx.x;
}



//! \todo use unsigned4 instead?
__device__ 
void 
rng_loadState(unsigned *rngState, const unsigned* g_rngState, size_t pitch)
{
	for(unsigned i=0; i<4; i++){
		rngState[i] = g_rngState[threadLocalStateIndex(i, pitch)];
	}
}


__device__ 
void 
rng_saveState(const unsigned *rngState, unsigned *g_rngState, size_t pitch)
{
	for(unsigned i=0; i<4; i++){
		g_rngState[threadLocalStateIndex(i, pitch)] = rngState[i];
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
        float* g_sigma,
        float* s_current)
{
	unsigned rngState[4];

	/* Copy the input state from memory into our local state */
	rng_loadState(rngState, g_rngState, pitch);
	
	for(uint nbase=0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {

		uint neuron = nbase + threadIdx.x;

		if(neuron < partitionSize) {

            size_t partitionOffset = CURRENT_PARTITION * pitch;
            //! \todo make use of  both randoms
            float2 r = rng_genGaussian(rngState);
            float sigma = g_sigma[partitionOffset + neuron];
            s_current[neuron] += r.x * sigma;
        }
    }
	
    /* Copy the current RNG state back to memory (not strictly necessary, you
     * can just generate a new random state every time if you want). */
	rng_saveState(rngState, g_rngState, pitch);
}
