#include "cycleCounting.cu_h"

#define CYCLE_COUNTING_THREAD 0

#ifdef KERNEL_TIMING
__shared__ clock_t s_cycleCounters[COUNTER_COUNT];
#endif

__device__
void
setCycleCounter(clock_t* s_cycleCounters, size_t counter)
{
	if(threadIdx.x == CYCLE_COUNTING_THREAD) {
		s_cycleCounters[counter] = (unsigned long long) clock();
	}
    __syncthreads();
}



__device__
void
writeCycleCounters(clock_t* s_cc, unsigned long long* g_cc, size_t ccPitch)
{
	__shared__ clock_t s_duration[DURATION_COUNT];

	__syncthreads();

	if(threadIdx.x == 0) {
		s_duration[DURATION_KERNEL]             = s_cc[COUNTER_KERNEL_END]                  - s_cc[COUNTER_KERNEL_START];
		s_duration[DURATION_KERNEL_INIT]        = s_cc[COUNTER_KERNEL_INIT_LOADL1]          - s_cc[COUNTER_KERNEL_START];
		s_duration[DURATION_KERNEL_LOADL1]      = s_cc[COUNTER_KERNEL_LOADL1_LOADFIRING]    - s_cc[COUNTER_KERNEL_INIT_LOADL1];
		s_duration[DURATION_KERNEL_LOADFIRING]  = s_cc[COUNTER_KERNEL_LOADFIRING_INTEGRATE] - s_cc[COUNTER_KERNEL_LOADL1_LOADFIRING];
		s_duration[DURATION_KERNEL_INTEGRATE]   = s_cc[COUNTER_KERNEL_INTEGRATE_FIRE]       - s_cc[COUNTER_KERNEL_LOADFIRING_INTEGRATE];
		s_duration[DURATION_KERNEL_FIRE]        = s_cc[COUNTER_KERNEL_FIRE_STOREL1]         - s_cc[COUNTER_KERNEL_INTEGRATE_FIRE];
		s_duration[DURATION_KERNEL_STOREL1]     = s_cc[COUNTER_KERNEL_STOREL1_STOREFIRING]  - s_cc[COUNTER_KERNEL_FIRE_STOREL1];
		s_duration[DURATION_KERNEL_STOREFIRING] = s_cc[COUNTER_KERNEL_END]                  - s_cc[COUNTER_KERNEL_STOREL1_STOREFIRING];
	}

	__syncthreads();

	if(threadIdx.x < DURATION_COUNT) {		
		atomicAdd(g_cc + blockIdx.x * ccPitch + threadIdx.x,
			(unsigned long long) s_duration[threadIdx.x]);
	}
}


#ifdef KERNEL_TIMING
#define SET_COUNTER(counter) setCycleCounter(s_cycleCounters, counter)
#define WRITE_COUNTERS(g_cc, ccPitch) writeCycleCounters(s_cycleCounters, g_cc, ccPitch)
#else
#define SET_COUNTER(counter)
#define WRITE_COUNTERS(g_cc, ccPitch)
#endif
