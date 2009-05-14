/*! \brief Run-time assertions on the GPU
 *
 * If the kernel is compiled with device assertions (CPP flag
 * DEVICE_ASSERTIONS), the kernel can perform run-time assertions, logging
 * location data to global memory. Only the line-number is recorded, so some
 * guess-work my be required to work out exactly what assertion failed. There
 * is only one assertion failure slot per thread, so it's possible to overwrite
 * an assertion failure.
 *
 * \author Andreas Fidjeland
 */

#include <cutil.h>
#include <stdio.h>
#include <vector>

#include "kernel.cu_h"


#ifdef DEVICE_ASSERTIONS

#define DEVICE_ASSERTION_MEMSZ (MAX_PARTITION_COUNT * THREADS_PER_BLOCK)

__device__ int g_assertions[DEVICE_ASSERTION_MEMSZ];


__device__ __host__
size_t
assertion_offset(size_t block, size_t thread)
{
    return block * THREADS_PER_BLOCK + thread;
}



#ifdef __DEVICE_EMULATION__
#	define ASSERT(cond) assert(cond)
#else
#	define ASSERT(cond) \
        if(!(cond)) {\
			g_assertions[assertion_offset(blockIdx.x, threadIdx.x)] = __LINE__;\
        }
#endif

#else // DEVICE_ASSERTIONS
#   define ASSERT(cond)
#endif



/* Check the assertion flags on the device and print the value (line numbers)
 * of all assertion failures.
 *
 * \return failure */
__host__
bool
assertionsFailed(size_t blocks, int cycle)
{
#ifdef DEVICE_ASSERTION_MEMSZ
	std::vector<int> h_assertions(DEVICE_ASSERTION_MEMSZ);
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(
            &h_assertions[0],
            g_assertions,
			DEVICE_ASSERTION_MEMSZ*sizeof(int), 0,
            cudaMemcpyDeviceToHost));

    bool failure = false;
    for(int block=0; block<blocks; ++block) {
        for(int thread=0; thread<THREADS_PER_BLOCK; ++thread) {
            int line = h_assertions[assertion_offset(block, thread)];
            if(line != 0) {
                fprintf(stderr, 
                    "Device assertion failure in block=%d, thread=%d, line=%d, cycle=%d\n",
                    block, thread, line, cycle);
                failure = true;
            }
        }
    }

    return failure;
#else
	return false;
#endif
}



__host__
void
clearAssertions()
{
#ifdef DEVICE_ASSERTIONS
	void* addr;
	CUDA_SAFE_CALL(cudaGetSymbolAddress(&addr, g_assertions));
	CUDA_SAFE_CALL(cudaMemset(addr, 0, DEVICE_ASSERTION_MEMSZ*sizeof(int)));
#endif
}
