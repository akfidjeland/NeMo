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


/* Network-wide configuration */

#define NPARAM_maxPartitionSize 0
#define NPARAM_maxDelay 1
#define NPARAM_pitch32 2
#define NPARAM_pitchL0 3
#define NPARAM_sizeL0 4
#define NPARAM_pitchL1 5
#define NPARAM_sizeL1 6
#define NPARAM_COUNT 7

/* Configuration array is stored in constant memory, and is loaded in
 * (parallel) into shared memory for each thread block */
__constant__ uint c_networkParameters[NPARAM_COUNT];
__shared__ uint s_networkParameters[NPARAM_COUNT];

/* Some more pleasant names for the parameters */
#define s_maxPartitionSize s_networkParameters[NPARAM_maxPartitionSize]
#define s_maxDelay s_networkParameters[NPARAM_maxDelay]
#define s_pitch32 s_networkParameters[NPARAM_pitch32]
#define s_pitchL0 s_networkParameters[NPARAM_pitchL0]
#define s_sizeL0 s_networkParameters[NPARAM_sizeL0]
#define s_pitchL1 s_networkParameters[NPARAM_pitchL1]
#define s_sizeL1 s_networkParameters[NPARAM_sizeL1]


#define SET_CONSTANT(symbol, val) param[NPARAM_ ## symbol] = val

__host__
void
configureKernel(RTDATA rtdata)
{
	std::vector<uint> param(NPARAM_COUNT);
	SET_CONSTANT(maxPartitionSize, rtdata->maxPartitionSize);
	SET_CONSTANT(maxDelay, rtdata->maxDelay());
	SET_CONSTANT(pitch32, rtdata->pitch32());
	SET_CONSTANT(pitchL0, rtdata->cm(CM_L0)->synapsePitchD());
	SET_CONSTANT(sizeL0, rtdata->cm(CM_L0)->submatrixSize());
	SET_CONSTANT(pitchL1, rtdata->cm(CM_L1)->synapsePitchD());
	SET_CONSTANT(sizeL1, rtdata->cm(CM_L1)->submatrixSize());
	CUDA_SAFE_CALL(
			cudaMemcpyToSymbol(c_networkParameters,
				&param[0], 
				sizeof(uint)*NPARAM_COUNT, 
				0,
				cudaMemcpyHostToDevice));
}


#define LOAD_CONSTANT(symbol) s_ ## symbol = c_ ## symbol

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
