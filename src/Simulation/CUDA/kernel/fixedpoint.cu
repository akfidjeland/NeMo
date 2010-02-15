#ifndef FIXED_POINT_CU
#define FIXED_POINT_CU

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
	

__device__
fix_t
fx_mul(fix_t a, fix_t b)
{
	ASSERT(sizeof(fix_t) == 4);
	int64_t r = int64_t(a) * int64_t(b);
	return fix_t(r >> c_fixedPointFractionalBits);
}


#endif
