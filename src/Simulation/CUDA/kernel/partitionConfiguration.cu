#ifndef PARTITION_CONFIGURATION_CU
#define PARTITION_CONFIGURATION_CU

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


#include <cutil.h>
#include "kernel.cu_h"
#include "kernel.h"
#include "RuntimeData.hpp"
#include "ConnectivityMatrix.hpp"


/* Network-wide configuration */

#define NPARAM_maxPartitionSize 0
#define NPARAM_maxDelay         1
#define NPARAM_pitch32          2
#define NPARAM_f0_pitch         3
#define NPARAM_f0_size          4
#define NPARAM_f1_pitch         5
#define NPARAM_f1_size          6
#define NPARAM_r0_pitch         7
#define NPARAM_r0_size          8
#define NPARAM_r1_pitch         9
#define NPARAM_r1_size         10
#define NPARAM_COUNT           11

/* Configuration array is stored in constant memory, and is loaded in
 * (parallel) into shared memory for each thread block */
__constant__ uint c_networkParameters[NPARAM_COUNT];
__shared__ uint s_networkParameters[NPARAM_COUNT];

/* Some more pleasant names for the parameters */
//! \todo auto-generate
#define s_maxPartitionSize s_networkParameters[NPARAM_maxPartitionSize]
#define s_maxDelay         s_networkParameters[NPARAM_maxDelay]
#define s_pitch32          s_networkParameters[NPARAM_pitch32]
#define sf0_pitch          s_networkParameters[NPARAM_f0_pitch]
#define sf0_size           s_networkParameters[NPARAM_f0_size]
#define sf1_pitch          s_networkParameters[NPARAM_f1_pitch]
#define sf1_size           s_networkParameters[NPARAM_f1_size]
#define sr0_pitch          s_networkParameters[NPARAM_r0_pitch]
#define sr0_size           s_networkParameters[NPARAM_r0_size]
#define sr1_pitch          s_networkParameters[NPARAM_r1_pitch]
#define sr1_size           s_networkParameters[NPARAM_r1_size]


#define SET_CONSTANT(symbol, val) param[NPARAM_ ## symbol] = val

__host__
void
configureKernel(RTDATA rtdata)
{
	std::vector<uint> param(NPARAM_COUNT);
	SET_CONSTANT(maxPartitionSize, rtdata->maxPartitionSize);
	SET_CONSTANT(maxDelay,  rtdata->maxDelay());
	SET_CONSTANT(pitch32,   rtdata->pitch32());
	SET_CONSTANT(f0_pitch,  rtdata->cm(CM_L0)->df_pitch());
	SET_CONSTANT(f0_size,   rtdata->cm(CM_L0)->df_planeSize());
	SET_CONSTANT(f1_pitch,  rtdata->cm(CM_L1)->df_pitch());
	SET_CONSTANT(f1_size,   rtdata->cm(CM_L1)->df_planeSize());
	SET_CONSTANT(r0_pitch,  rtdata->cm(CM_L0)->dr_pitch());
	SET_CONSTANT(r0_size,   rtdata->cm(CM_L0)->dr_planeSize());
	SET_CONSTANT(r1_pitch,  rtdata->cm(CM_L1)->dr_pitch());
	SET_CONSTANT(r1_size,   rtdata->cm(CM_L1)->dr_planeSize());
	CUDA_SAFE_CALL(
			cudaMemcpyToSymbol(c_networkParameters,
				&param[0], 
				sizeof(uint)*NPARAM_COUNT, 
				0,
				cudaMemcpyHostToDevice));
}


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

__constant__ uint cf0_maxSynapsesPerDelay[MAX_THREAD_BLOCKS];
__constant__ uint cr0_maxSynapsesPerNeuron[MAX_THREAD_BLOCKS];
__constant__ uint cf1_maxSynapsesPerDelay[MAX_THREAD_BLOCKS];
__constant__ uint cr1_maxSynapsesPerNeuron[MAX_THREAD_BLOCKS];

template<class T>
__host__
void
configurePartition(const T& symbol, const std::vector<uint>& values)
{
	std::vector<int> buf(MAX_THREAD_BLOCKS, 0);
	std::copy(values.begin(), values.end(), buf.begin());
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(
			symbol, &buf[0],
			MAX_THREAD_BLOCKS*sizeof(uint),
			0, cudaMemcpyHostToDevice));
}


__constant__ uint c_partitionSize[MAX_THREAD_BLOCKS];

__host__
void
configurePartitionSize(size_t n, const uint* d_partitionSize)
{
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(
			c_partitionSize, d_partitionSize,
			MAX_THREAD_BLOCKS*sizeof(uint), 
			0, cudaMemcpyHostToDevice));
}

#endif
