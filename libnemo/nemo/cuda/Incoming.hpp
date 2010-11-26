#ifndef INCOMING_HPP
#define INCOMING_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_ptr.hpp>

#include "incoming.cu_h"

namespace nemo {
	namespace cuda {

/*! \brief Queue for exchanging spike data between partitions
 *
 * This queue is used in the \ref cuda_global_delivery "global delivery step".
 * It is a 2D grid of queues with one entry per source partition/target
 * partition pair. Each individual queue contains warp indices and is sized
 * pessimistically so as to support continous firing of all neurons. The total
 * size of this data structure is thus \e pcount x \e pcount x \e max_incoming_warps.
 *
 * The actual queue data and the fill rate of each individual queue are stored
 * in separate data structures on the device.
 *
 * The device functions for manipulating queue data are found in \ref incoming.cu.
 *
 * \see scatterGlobal, gather, cuda_global_delivery
 */
class Incoming
{
	public :

		Incoming();

		/*! Allocate space on device to hold the per neuron/delay incoming
		 * spike groups
		 *
		 * \param partitionCount
		 * 		Number of partitions in whole network
		 * \param maxIncomingWarps
		 * 		Maximum number of incoming warps (regardless of delay) for any
		 * 		partition,
		 * \param sizeMultiplier
		 * 		To be completely safe against buffer overflow, base incoming
		 * 		buffer sizing on the assumption that all neurons may fire
		 * 		continously for some time. This is unlikely to happen in
		 * 		practice, however, so we can relax this. The size multiplier
		 * 		specifies how large the buffer should be wrt the most
		 * 		conservative case.
		 */
		void allocate(size_t partitionCount,
				size_t maxIncomingWarps,
				double sizeMultiplier = 1.0);

		incoming_t* buffer() const { return m_buffer.get(); }

		unsigned* heads() const { return m_count.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

	private :

		/* On the device there a buffer for incoming spike group for each
		 * (target) partition */
		boost::shared_ptr<incoming_t> m_buffer;

		/* At run-time, we keep track of how many incoming spike groups are
		 * queued for each target partition */
		boost::shared_ptr<unsigned> m_count;

		size_t m_allocated;
};

	} // end namespace cuda
} // end namespace nemo

#endif
