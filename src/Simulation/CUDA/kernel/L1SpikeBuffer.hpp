#ifndef INCOMING_HPP
#define INCOMING_HPP

#include <boost/shared_ptr.hpp>

#include "l1SpikeBuffer.cu_h"

class Incoming
{
	public :

		// default ctor is fine here

		void allocate(size_t partitionCount);

		incoming_t* buffer() const { return m_buffer.get(); }

		uint* heads() const { return m_count.get(); }

	private :

		/* On the device there a buffer for incoming spike group for each
		 * (target) partition */
		boost::shared_ptr<incoming_t> m_buffer;

		/* At run-time, we keep track of how many incoming spike groups are
		 * queued for each target partition */
		boost::shared_ptr<uint> m_count;
};

#endif
