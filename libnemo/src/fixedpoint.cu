#ifndef FIXED_POINT_CU
#define FIXED_POINT_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "device_assert.cu"
#include "cuda_types.h"
#include "cycle.cu"

#define FX_SIGN_BIT 0x80000000

/* Scaling factor used for fixed-points (used for weight storage) */
__constant__ unsigned c_fixedPointScale;
__constant__ unsigned c_fixedPointFractionalBits;


__host__
cudaError
fx_setFormat(unsigned fracbits)
{
	unsigned scale = 1 << fracbits;
	cudaError status;
	status = cudaMemcpyToSymbol(c_fixedPointScale,
				&scale, sizeof(unsigned), 0, cudaMemcpyHostToDevice);
	if(cudaSuccess != status) {
		return status;
	}
	return cudaMemcpyToSymbol(c_fixedPointFractionalBits,
				&fracbits, sizeof(unsigned), 0, cudaMemcpyHostToDevice);
}


__device__
float
fx_tofloat(fix_t v)
{
	//! \todo any way to avoid division here. Perhaps precompute the fraction here?
	//! \todo check if it makes any difference to use 1<<c here instead
	return float(v) / c_fixedPointScale;
}



/*! \return saturated value with the given sign (bit 0 in 'sign') */
__device__
fix_t
fx_saturate(bool negative)
{
	return fix_t(~0) & (fix_t(negative) << 31);
}


__device__
float
fx_saturatedTofloat(fix_t v, bool overflow, bool negative)
{
	//! \todo any way to avoid division here. Perhaps precompute the fraction here?
	//! \todo check if it makes any difference to use 1<<c here instead
	return float(overflow ? fx_saturate(negative) : v) / c_fixedPointScale;
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


__device__
fix_t
fx_isNegative(fix_t v)
{
	return v & FX_SIGN_BIT;
}



/* Convert shared-memory array from fixed-point to floating point format and
 * perform fixed-point saturation. The conversion can be done in-place, i.e.
 * the fixed-point input and floating-point outputs arrays can be the same. */
__device__
void
fx_arrSaturatedToFloat(
		uint32_t* s_overflow, // bit-vector
		uint32_t* s_negative, // bit-vector
		fix_t* s_fix,
		float* s_float)
{
	/* If any accumulators overflow, clamp to max positive or minimum value */
	for(unsigned nbase=0; nbase < MAX_PARTITION_SIZE; nbase += THREADS_PER_BLOCK) {
		unsigned nidx = nbase + threadIdx.x;
#ifndef FIXPOINT_SATURATION
		s_float[nidx] = fx_tofloat(s_fix[nidx]);
#else
		bool overflow = bv_isSet(nidx, s_overflow);
		bool negative = bv_isSet(nidx, s_negative);
		s_float[nidx] = fx_saturatedTofloat(s_fix[nidx], overflow, negative);
		if(overflow) {
			DEBUG_MSG("c%u p%un%u input current overflow. Saturated to %+f (%08x)\n",
					s_cycle, CURRENT_PARTITION, nidx,
					fx_tofloat(s_fix[nidx]), s_fix[nidx]);
		}
#endif
	}
	__syncthreads();
}





#endif
