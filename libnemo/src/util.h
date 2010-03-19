#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdint.h>

#define IS_POWER_OF_TWO(v) (!((v) & ((v) - 1)) && (v))

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))

/* Round a up to the nearest multiple of b */
#define ALIGN(a, b) (b) * DIV_CEIL((a), (b))

#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))

//! \todo report errors back to user of library
//! \move this to cpp code and make use of exceptions
#define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                  \
    if( cudaSuccess != err) {                                              \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                __FILE__, __LINE__, cudaGetErrorString( err) );            \
        exit(EXIT_FAILURE);                                                \
    } }

#define MASK(bits) (~(~0 << (bits)))

#endif
