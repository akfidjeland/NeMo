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
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {

Incoming::Incoming() : m_allocated(0) {}


void
Incoming::allocate(size_t partitionCount, size_t maxIncomingWarps, double sizeMultiplier)
{
	// allocate space for the incoming count (double-buffered)
	unsigned* d_count;
	size_t len = ALIGN(partitionCount * 2, 32) * sizeof(unsigned);
	d_malloc((void**)&d_count, len, "incoming spike queue counts");
	m_count = boost::shared_ptr<unsigned>(d_count, d_free);
	d_memset(d_count, 0, len);
	m_allocated = len;

	/* The queue has one entry for incoming spikes for each partition */
	assert(partitionCount < MAX_PARTITION_COUNT);
	size_t height = partitionCount * 2; // double buffered

	/* Each buffer entry (for a particular target partition) is of a fixed size.
	 * The sizing of this is very conservative. In fact the buffer is large
	 * enough that every neuron can fire every cycle. */
	/*! \todo relax this constraint. We'll end up using a very large amount of
	 * space when using a large number of partitions */
	assert(sizeMultiplier > 0.0);
	double mult = std::min(1.0, sizeMultiplier);
	size_t width = size_t(mult * maxIncomingWarps * sizeof(incoming_t));

	incoming_t* d_buffer;
	size_t bpitch;

	d_mallocPitch((void**)&d_buffer, &bpitch, width, height, "incoming spike queue");
	m_allocated += bpitch * height;

	m_buffer = boost::shared_ptr<incoming_t>(d_buffer, d_free);

	/* We don't need to clear the queue. It will generally be full of garbage
	 * anyway. The queue heads must be used to determine what's valid data */

	size_t wpitch = bpitch / sizeof(incoming_t);
	CUDA_SAFE_CALL(setIncomingPitch(wpitch));
}

	} // end namespace cuda
} // end namespace nemo
