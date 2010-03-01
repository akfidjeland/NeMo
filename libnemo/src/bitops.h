#ifndef BITOPS_H
#define BITOPS_H

#include <limits.h>

/* Compute leading zeros for type T which should have B bits.
 *
 * This could be done faster using one of the other bit-twiddling hacks from
 * http://graphics.stanford.edu/~seander/bithacks.html */
template<typename T, int B>
int
clzN(T v)
{
	uint r = 0;
	while (v >>= 1) {
		r++;
	}
	return (B - 1) - r;
}


/* Count leading zeros in 64-bit word. Unfortunately the gcc builtin to deal
 * with this is not explicitly 64 bit. Instead it is defined for long long. In
 * C99 this is required to be /at least/ 64 bits. However, we require it to be
 * /exactly/ 64 bits. */
#if LLONG_MAX == 9223372036854775807
inline int clz64(uint64_t val) { return __builtin_clzll(val); }
#else
#warning "long long is not 64 bit, using slow clzll"
inline int clz64(uint64_t val) { return clzN<uint64_t, 64>(val); }
#endif // LLONG_MAX


/* Ditto for 32 bits */
#if UINT_MAX == 4294967295U
inline int clz32(uint32_t val) { return __builtin_clz(val); }
#else
#warning "long int is not 32 bit, using slow clzl"
inline int clz32(uint32_t val) { return clzN<uint32_t, 32>(val); }
#endif // LONG_MAX


/* Count trailing zeros. This should work even if long long is greater than
 * 64-bit. The uint64_t will be safely extended to the appropriate length */
inline int ctz64(uint64_t val) { return __builtin_ctzll(val); }


#endif // BITOPS_H
