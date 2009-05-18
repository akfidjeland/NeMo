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


/* Kernel-wide configuration */

__constant__ uint c_maxPartitionSize;
__constant__ uint c_maxDelay;


__host__
void
configureKernel(uint maxPartitionSize, uint maxDelay)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_maxPartitionSize, 
				&maxPartitionSize, sizeof(uint), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_maxDelay, 
				&maxDelay, sizeof(uint), 0, cudaMemcpyHostToDevice));
}


/* Per-partition configuration */

__constant__ int c_maxL0SynapsesPerDelay    [MAX_THREAD_BLOCKS];
__constant__ int c_maxL0RevSynapsesPerDelay [MAX_THREAD_BLOCKS];
__constant__ int c_maxL1SynapsesPerDelay    [MAX_THREAD_BLOCKS];
__constant__ int c_maxL1RevSynapsesPerDelay [MAX_THREAD_BLOCKS];

template<class T>
__host__
void
configurePartition(const T& symbol, const std::vector<int>& values)
{
	std::vector<int> buf(MAX_THREAD_BLOCKS, 0);
	std::copy(values.begin(), values.end(), buf.begin());
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(
			symbol, &buf[0],
			MAX_THREAD_BLOCKS*sizeof(int),
			0, cudaMemcpyHostToDevice));
}


__constant__ int c_partitionSize[MAX_THREAD_BLOCKS];

__host__
void
configurePartitionSize(size_t n, const int* d_partitionSize)
{
	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(
			c_partitionSize, d_partitionSize,
			MAX_THREAD_BLOCKS*sizeof(int), 
			0, cudaMemcpyHostToDevice));
}
