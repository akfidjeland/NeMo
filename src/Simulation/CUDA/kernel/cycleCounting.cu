#include "cycleCounting.cu_h"

/* Each kernel has a separate cycle counter */
#ifdef KERNEL_TIMING
__shared__ clock_t s_ccMain[CC_MAIN_COUNT];
//! \todo don't allocate this memory if STDP not enabled
__shared__ clock_t s_ccReorderSTDP[CC_STDP_REORDER_COUNT];
__shared__ clock_t s_ccApplySTDP[CC_STDP_APPLY_COUNT];
#endif


/* Calculate the duration based on start and end times. The clock counters are
 * 32-bit and silently wrap around, at least on G80. It seems that clock_t is
 * not neccessarily 32-bit, so we need a bit of hard-coding here. This is not
 * future-proof, obviously. */
__device__
clock_t
duration(clock_t start, clock_t end)
{
	if (end > start)
		return end - start;
	else
		return end + (0xffffffff - start);
}

__device__
void
setCycleCounter(clock_t* s_cc, size_t counter)
{
	if(threadIdx.x == 0) {
		s_cc[counter] = clock();
	}
}



__device__
void
writeCycleCounters(clock_t* s_cc, unsigned long long* g_cc, size_t pitch, size_t count)
{
    __syncthreads();
	if(threadIdx.x < count-1) {
		clock_t d = duration(s_cc[threadIdx.x], s_cc[threadIdx.x+1]);
		atomicAdd(g_cc + blockIdx.x * pitch + threadIdx.x, (unsigned long long) d);
	}
}


#ifdef KERNEL_TIMING
//! \todo add separate methods for start and end counters?
#define SET_COUNTER(s_cc, counter) setCycleCounter(s_cc, counter)
#define WRITE_COUNTERS(s_cc, g_cc, ccPitch, ccCount) writeCycleCounters(s_cc, g_cc, ccPitch, ccCount)
#else
#define SET_COUNTER(s_cc, counter)
#define WRITE_COUNTERS(s_cc, g_cc, ccPitch, ccCount)
#endif
