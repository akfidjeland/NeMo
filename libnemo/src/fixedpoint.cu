#ifndef FIXED_POINT_CU
#define FIXED_POINT_CU

#include "error.cu"
#include "util.h"
#include "nemo_cuda_types.h"

#define FX_SIGN_BIT 0x80000000

/* Scaling factor used for fixed-points (used for weight storage) */
__constant__ uint c_fixedPointScale;
__constant__ uint c_fixedPointFractionalBits;


__host__
void
setFixedPointFormat(uint fracbits)
{
	uint scale = 1 << fracbits;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_fixedPointScale,
				&scale, sizeof(uint), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_fixedPointFractionalBits,
				&fracbits, sizeof(uint), 0, cudaMemcpyHostToDevice));
}


__device__
float
fx_tofloat(fix_t v)
{
	//! \todo any way to avoid division here. Perhaps precompute the fraction here?
	//! \todo check if it makes any difference to use 1<<c here instead
	return float(v) / c_fixedPointScale;
}



/*! Add atomically to shared memory fixed-point value, returning true if an
 * overflow occurred */
__device__
bool
fx_atomicAdd(fix_t* s_a, fix_t b)
{
	fix_t a = atomicAdd(s_a, b);
	/* It seems it's not possible to access the carry bit, even in PTX code.
	 * (from PTX manual under ADDC). We therefore have to manually check for
	 * overflow. */
	fix_t aSign = a & FX_SIGN_BIT;
	fix_t bSign = b & FX_SIGN_BIT;
	/* We cannot rely on *s_a here, due to race conditions */
	fix_t outputSign = (a+b) & FX_SIGN_BIT;
	/*! \note could use 64-bit addition here for overflow detection */
	return (aSign == bSign) && (aSign != outputSign);
}


__device__
fix_t
fx_mul(fix_t a, fix_t b)
{
	ASSERT(sizeof(fix_t) == 4);
	int64_t r = int64_t(a) * int64_t(b);
	return fix_t(r >> c_fixedPointFractionalBits);
}


/*! \return saturated value with the given sign (bit 0 in 'sign') */
__device__
fix_t
fx_saturate(bool negative)
{
	return fix_t(~0) & (fix_t(negative) << 31);
}


__device__
fix_t
fx_isNegative(fix_t v)
{
	return v & FX_SIGN_BIT;
}



#endif
