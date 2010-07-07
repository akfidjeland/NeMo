/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Incoming.hpp"

#include <nemo/util.h>

#include "kernel.cu_h"
#include "exception.hpp"

namespace nemo {
	namespace cuda {

Incoming::Incoming() : m_allocated(0) {}


void
Incoming::allocate(size_t partitionCount, size_t maxIncomingWarps, double sizeMultiplier)
{
	// allocate space for the incoming count
	unsigned* d_count;
	size_t len = ALIGN(partitionCount * MAX_DELAY, 32) * sizeof(unsigned);
	cudaError err = cudaMalloc((void**)&d_count, len);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("incoming spike queue counts", len, err);
	}
	m_count = boost::shared_ptr<unsigned>(d_count, cudaFree);

	err = cudaMemset(d_count, 0, len);
	if(cudaSuccess != err) {
		throw nemo::exception(NEMO_CUDA_MEMORY_ERROR, "failed to set incoming spike queue counts");
	}

	m_allocated = len;

	/* The queue has one entry for incoming spikes for each partition */
	assert(partitionCount < MAX_PARTITION_COUNT);
	size_t height = partitionCount * MAX_DELAY;

	/* Each buffer entry (for a particular source partition) is of a fixed size
	 * to simplify the rotating buffer code. This is very conservative. In fact
	 * the buffer is large enough that every neuron can fire every cycle */
	/*! \todo relax this constraint. We'll end up using a very large amount of
	 * space when using a large number of partitions */
	assert(sizeMultiplier > 0.0);
	double mult = std::min(1.0, sizeMultiplier);
	size_t width = size_t(mult * maxIncomingWarps * sizeof(incoming_t));

	incoming_t* d_buffer;
	size_t bpitch;

	err = cudaMallocPitch((void**)&d_buffer, &bpitch, width, height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("incoming spike queue", width * height, err);
	}
	m_allocated += bpitch * height;

	m_buffer = boost::shared_ptr<incoming_t>(d_buffer, cudaFree);

	/* We don't need to clear the queue. It will generally be full of garbage
	 * anyway. The queue heads must be used to determine what's valid data */

	size_t wpitch = bpitch / sizeof(incoming_t);
	CUDA_SAFE_CALL(setIncomingPitch(wpitch));
}

	} // end namespace cuda
} // end namespace nemo
