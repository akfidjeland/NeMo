#ifndef UTIL_H
#define UTIL_H

#define IS_POWER_OF_TWO(v) (!((v) & ((v) - 1)) && (v))

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

/* Round a up to the nearest multiple of b */
#define ALIGN(a, b) (b) * DIV_CEIL((a), (b))

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

#define MASK(bits) (~(~0 << (bits)))

#endif
