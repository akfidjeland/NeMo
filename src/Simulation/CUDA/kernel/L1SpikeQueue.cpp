#include "L1SpikeQueue.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cutil.h>
#include <assert.h>


L1SpikeQueue::L1SpikeQueue(
        size_t partitionCount,
		size_t entrySize) :
	m_data(NULL),
	m_pitch(0),
	m_heads(NULL),
	m_headPitch(0)
{
	if(partitionCount != 1 && entrySize != 0) {
		{
			/* We need a double buffer here, so we can read and write concurrently. */
			const size_t height = partitionCount * partitionCount * 2;
			const size_t width  = entrySize * sizeof(uint2);
			size_t bpitch = 0;
			CUDA_SAFE_CALL(cudaMallocPitch(
						(void**)&m_data, &bpitch, width, height));
			cudaMemset2D(m_data, bpitch, 0x0, bpitch, height);
			assert(bpitch != 0);
			m_pitch = bpitch / sizeof(uint2);
		}

		{
			const size_t height = partitionCount * 2;
			const size_t width  = partitionCount * sizeof(unsigned int);
			size_t bpitch;
			CUDA_SAFE_CALL(cudaMallocPitch((void**)&m_heads,
						&bpitch, width, height));
			assert(bpitch != 0);
			cudaMemset2D(m_heads, bpitch, 0x0, bpitch, height);
			m_headPitch = bpitch / sizeof(unsigned int);
		}
	}
}


L1SpikeQueue::~L1SpikeQueue()
{
	cudaFree(m_data);
	cudaFree(m_heads);
}


uint2*
L1SpikeQueue::data() const
{
	return m_data;
}


size_t
L1SpikeQueue::pitch() const
{
	return m_pitch;
}


unsigned int*
L1SpikeQueue::heads() const
{
	return m_heads;
}


size_t
L1SpikeQueue::headPitch() const
{
	return m_headPitch;
}
