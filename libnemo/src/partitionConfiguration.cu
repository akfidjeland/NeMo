#ifndef PARTITION_CONFIGURATION_CU
#define PARTITION_CONFIGURATION_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


//-----------------------------------------------------------------------------
// Kernel configuration
//-----------------------------------------------------------------------------
//
// Different clusters may have slightly different configuration, e.g. with
// respect to what external inputs they require. This information is all kept
// in constant memory and is thus cached. 
// 
// It is not possible to dynamically size data structures in constant memory,
// so we simply set some upper limit on the number of thread blocks
// (MAX_THREAD_BLOCKS) and size the data structures statically.
//-----------------------------------------------------------------------------

#include "kernel.cu_h"


/* Network-wide configuration */

//! \todo reorder
#define NPARAM_maxDelay         1
#define NPARAM_pitch32          2
#define NPARAM_pitch64          3
#define NPARAM_COUNT            4

/* Configuration array is stored in constant memory, and is loaded in
 * (parallel) into shared memory for each thread block */
__constant__ unsigned c_networkParameters[NPARAM_COUNT];
__shared__ unsigned s_networkParameters[NPARAM_COUNT];

/* Some more pleasant names for the parameters */
// s_maxDelay is not currently in use, and could potentially be removed
#define s_maxDelay         s_networkParameters[NPARAM_maxDelay]
#define s_pitch32          s_networkParameters[NPARAM_pitch32]
#define s_pitch64          s_networkParameters[NPARAM_pitch64]


#define SET_CONSTANT(symbol, val) param[NPARAM_ ## symbol] = val

//! \todo split this up
__host__
cudaError
configureKernel(unsigned maxDelay, unsigned pitch32, unsigned pitch64)
{
	unsigned param[NPARAM_COUNT];
	SET_CONSTANT(maxDelay,  maxDelay);
	SET_CONSTANT(pitch32,   pitch32);
	SET_CONSTANT(pitch64,   pitch64);
	return cudaMemcpyToSymbol(c_networkParameters,
				&param[0],
				sizeof(unsigned)*NPARAM_COUNT,
				0,
				cudaMemcpyHostToDevice);
}


//! \todo just use constant memory directly
__device__
void
loadNetworkParameters()
{
	if(threadIdx.x < NPARAM_COUNT) {
		s_networkParameters[threadIdx.x] = c_networkParameters[threadIdx.x];
	}
	__syncthreads();
}


/* Per-partition configuration */

__constant__ unsigned c_partitionSize[MAX_THREAD_BLOCKS];

__host__
cudaError
configurePartitionSize(const unsigned* d_partitionSize, size_t len)
{
	//! \todo set padding to zero
	assert(len <= MAX_THREAD_BLOCKS);
	return cudaMemcpyToSymbol(
			c_partitionSize,
			(void*) d_partitionSize,
			MAX_THREAD_BLOCKS*sizeof(unsigned),
			0, cudaMemcpyHostToDevice);
}

#endif
