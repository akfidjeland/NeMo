#include "firingProbe.cu_h"
#include "kernel.cu_h"
#include "util.h"

/*! Update the firing output buffer in parallel and possibly split over
 * multiple chunks.
 *
 * \param g_nextFree
 *		Global memory containing per-partition entries containing the word
 *		offset of the next available slot for firing.
 * \param s_firing
 *		Vector of neuron indices of fired neurons
 * \param firingCount
 *		Number of firing data in shared memory buffer
 * \param g_firing
 *		Output buffer in global memory, \see FiringProbe
 */
__device__
void
writeFiringOutput(
	ushort cycle,
	uint* g_nextFree,
	uint16_t* s_firingIdx,
	uint firingCount,
	ushort2* g_firing)
{
	__shared__ uint s_nextFree; 
	__shared__ uint s_loopIterations;

	/* The host reads processes a whole number of chunks, so we must write at
	 * least up to the nearest chunk boundary */ 
	__shared__ uint s_maxWriteOffset;

	if(threadIdx.x == 0) {
		/* We already know how much firing to write, so we can update this in
		 * one go */	
		int chunksToWrite = (firingCount + FMEM_CHUNK_SIZE - 1) / FMEM_CHUNK_SIZE;
		s_nextFree = atomicAdd(g_nextFree + blockIdx.x, 
				chunksToWrite * PARTITION_COUNT * FMEM_CHUNK_SIZE);
		s_loopIterations = (firingCount + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
		s_maxWriteOffset = ALIGN(firingCount, FMEM_CHUNK_SIZE);
		//! \todo check for buffer overflow
	}
	__syncthreads();

	/* loop to deal with more neurons than threads */
	for(int i=0; i<s_loopIterations; ++i) {
		size_t s_offset = i * THREADS_PER_BLOCK + threadIdx.x;
		if(s_offset < s_maxWriteOffset) {
			int chunk = s_offset / FMEM_CHUNK_SIZE;
			int chunkOffset = s_offset % FMEM_CHUNK_SIZE;
			/* Need to set unused entries to known value, to avoid leaving
			 * garbage for the host to later mis-interpret */
			short firingIdx = s_offset < firingCount ? 
				s_firingIdx[s_offset] :
				INVALID_NEURON;
			size_t g_offset = s_nextFree + chunk * PARTITION_COUNT * FMEM_CHUNK_SIZE + chunkOffset;
			g_firing[g_offset] = make_ushort2(cycle, firingIdx);
		}
	}	
}
