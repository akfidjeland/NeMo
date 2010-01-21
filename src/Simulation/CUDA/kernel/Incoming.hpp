#ifndef INCOMING_HPP
#define INCOMING_HPP

#include <boost/shared_ptr.hpp>

#include "incoming.cu_h"

class Incoming
{
	public :

		Incoming();

		/*! Allocate space on device to hold the per neuron/delay incoming
		 * spike groups
		 *
		 * \param maxIncomingWarps
		 * 		Maximum number of incoming warps (regardless of delay) for any
		 * 		partition,
		 */
		void allocate(size_t partitionCount, size_t maxIncomingWarps);

		incoming_t* buffer() const { return m_buffer.get(); }

		uint* heads() const { return m_count.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

	private :

		/* On the device there a buffer for incoming spike group for each
		 * (target) partition */
		boost::shared_ptr<incoming_t> m_buffer;

		/* At run-time, we keep track of how many incoming spike groups are
		 * queued for each target partition */
		boost::shared_ptr<uint> m_count;

		size_t m_allocated;
};

#endif
