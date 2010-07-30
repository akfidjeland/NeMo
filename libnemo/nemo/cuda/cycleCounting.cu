#ifndef CYCLE_COUNTING_CU
#define CYCLE_COUNTING_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"

/* Each kernel has a separate cycle counter */
#ifdef NEMO_CUDA_KERNEL_TIMING
__shared__ clock_t s_ccMain[CC_MAIN_COUNT];
//! \todo don't allocate this memory if STDP not enabled
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


#ifdef NEMO_CUDA_KERNEL_TIMING
//! \todo add separate methods for start and end counters?
#define SET_COUNTER(s_cc, counter) setCycleCounter(s_cc, counter)
#define WRITE_COUNTERS(s_cc, g_cc, ccPitch, ccCount) writeCycleCounters(s_cc, g_cc, ccPitch, ccCount)
#else
#define SET_COUNTER(s_cc, counter)
#define WRITE_COUNTERS(s_cc, g_cc, ccPitch, ccCount)
#endif

#endif
