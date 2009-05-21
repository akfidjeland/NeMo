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
__constant__ size_t c_pitchL0;
__constant__ size_t c_sizeL0;
__constant__ size_t c_pitchL1;
__constant__ size_t c_sizeL1;


#define SET_CONSTANT(symbol) CUDA_SAFE_CALL(\
		cudaMemcpyToSymbol(c_ ## symbol, &symbol, sizeof(symbol), 0, cudaMemcpyHostToDevice)\
	)


__host__
void
configureKernel(
		uint maxPartitionSize,
		uint maxDelay,
		size_t pitchL0,
		size_t sizeL0,
		size_t pitchL1,
		size_t sizeL1)
{
	SET_CONSTANT(maxPartitionSize);
	SET_CONSTANT(maxDelay);
	SET_CONSTANT(pitchL0);
	SET_CONSTANT(sizeL0);
	SET_CONSTANT(pitchL1);
	SET_CONSTANT(sizeL1);
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
