#ifndef BITOPS_H
#define BITOPS_H

#include <limits.h>

/* Count leading zeros in 64-bit word. Unfortunately the gcc builtin to deal
 * with this is not explicitly 64 bit. Instead it is defined for long long. In
 * C99 this is required to be /at least/ 64 bits. However, we require it to be
 * /exactly/ 64 bits. */

#if LLONG_MAX == 9223372036854775807

inline int clz64(uint64_t val) { return __builtin_clzll(val); }

#else

#warning "long long is not 64 bit, using slow clzll"

/* This could be done faster using one of the other bit-twiddling hacks from
 * http://graphics.stanford.edu/~seander/bithacks.html */
inline
int
clz64(uint64_t v)
{
	uint r = 0;
	while (v >>= 1) {
		r++;
	}
	return 63 - r;
}

#endif // LLONG_MAX

/* Count trailing zeros. This should work even if long long is greater than
 * 64-bit. The uint64_t will be safely extended to the appropriate length */
inline int ctz64(uint64_t val) { return __builtin_ctzll(val); }


#endif // BITOPS_H
