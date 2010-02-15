#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>

#define IS_POWER_OF_TWO(v) (!((v) & ((v) - 1)) && (v))

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

/* Round a up to the nearest multiple of b */
#define ALIGN(a, b) (b) * DIV_CEIL((a), (b))

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

//! \todo report errors back to user of library
#define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                  \
    if( cudaSuccess != err) {                                              \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                __FILE__, __LINE__, cudaGetErrorString( err) );            \
        exit(EXIT_FAILURE);                                                \
    } }

#define MASK(bits) (~(~0 << (bits)))


/* compute the next highest power of 2 of 32-bit v. From "bit-twiddling hacks".  */
//! \todo merge with code in bitops.h
inline
uint32_t
ceilPowerOfTwo(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


#endif
