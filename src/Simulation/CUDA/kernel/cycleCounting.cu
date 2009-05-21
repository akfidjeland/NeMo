#include "cycleCounting.cu_h"

#ifdef KERNEL_TIMING
__shared__ clock_t s_cycleCounters[COUNTER_COUNT];
#endif

__device__
void
setCycleCounter(clock_t* s_cycleCounters, size_t counter)
{
	if(threadIdx.x == 0) {
		s_cycleCounters[counter] = (unsigned long long) clock();
	}
}



__device__
void
writeCycleCounters(clock_t* s_cc, unsigned long long* g_cc, size_t ccPitch)
{
    __syncthreads();
	if(threadIdx.x < DURATION_COUNT-1) {
		clock_t duration = s_cc[threadIdx.x+1] - s_cc[threadIdx.x];
		atomicAdd(g_cc + blockIdx.x * ccPitch + threadIdx.x,
			(unsigned long long) duration);
	}
}


#ifdef KERNEL_TIMING
#define SET_COUNTER(counter) setCycleCounter(s_cycleCounters, counter)
#define WRITE_COUNTERS(g_cc, ccPitch) writeCycleCounters(s_cycleCounters, g_cc, ccPitch)
#else
#define SET_COUNTER(counter)
#define WRITE_COUNTERS(g_cc, ccPitch)
#endif
