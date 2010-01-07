#include "L1SpikeBuffer.hpp"

#include "util.h"
#include "kernel.cu_h"


void
L1SpikeBuffer::allocate(size_t partitionCount)
{
	// allocate space for the queue heads
	uint* d_heads;
	size_t len = ALIGN(partitionCount, 32) * sizeof(uint);
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_heads, len));
	CUDA_SAFE_CALL(cudaMemset(d_heads, 0, len));
	m_heads = boost::shared_ptr<uint>(d_heads, cudaFree);

	/* The queue has one entry for incoming spikes for each partition */
	assert(partitionCount < MAX_PARTITION_COUNT);
	size_t height = partitionCount * MAX_DELAY;

	/* We're extremely conservative in the sizing of each buffer: it can
	 * support every neuron firing every cycle. */
	/*! \todo relax this constraint. We'll end up using a very large amount of
	 * space when using a large number of partitions */
	size_t width = partitionCount * MAX_PARTITION_SIZE;

	l1spike_t* d_buffer;
	size_t pitch;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_buffer, &pitch, width, height));
	m_buffer = boost::shared_ptr<l1spike_t>(d_buffer);

	/* We don't need to clear the queue. It will generally be full of garbage
	 * anyway. The queue heads must be used to determine what's valid data */

	setBufferPitch(pitch);
}
