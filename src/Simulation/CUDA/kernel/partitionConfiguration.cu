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
