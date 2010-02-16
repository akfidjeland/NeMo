#ifndef BIT_VECTOR_CU
#define BIT_VECTOR_CU

#include "bitvector.cu_h"


__constant__ size_t c_bv_pitch;


/*! Set common pitch for bitvectors */
__host__
void
bv_setPitch(size_t pitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_bv_pitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}


/*! Clear whole bitvector */
__device__
void
bv_clear(uint32_t* s_vec)
{
	ASSERT(THREADS_PER_BLOCK >= c_bv_pitch);
	if(threadIdx.x < c_bv_pitch) {
		s_vec[threadIdx.x] = 0;
	}
}


/*! Clear whole bitvector */
__device__
void
bv_clear_(uint32_t* s_vec)
{
	bv_clear(s_vec);
	__syncthreads();
}


/*! Check if a particular bit is set */
__device__
bool
bv_isSet(nidx_t neuron, uint32_t* s_vec)
{
	return (s_vec[neuron/32] >> (neuron % 32)) & 0x1;
}



/*! Set bit vector for \a neuron */
__device__
void
bv_atomicSet(nidx_t neuron, uint32_t* s_vec)
{
	uint word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	atomicOr(s_vec + word, mask);
}



/*! Set bit vector for \a neuron given that \a condition is true */
__device__
void
bv_atomicSetPredicated(bool condition, nidx_t neuron, uint32_t* s_vec)
{
	uint word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	if(condition) {
		atomicOr(s_vec + word, mask);
	}
}


/*! Copy bit vector */
__device__
void
bv_copy(uint32_t* src, uint32_t* dst)
{
	ASSERT(THREADS_PER_BLOCK >= c_bv_pitch);
	if(threadIdx.x < c_bv_pitch) {
		dst[threadIdx.x] =  src[threadIdx.x];
	}
}


#endif
