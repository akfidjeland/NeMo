#ifndef L1_SPIKE_BUFFER_HPP
#define L1_SPIKE_BUFFER_HPP

#include <boost/shared_ptr.hpp>

#include "l1SpikeBuffer.cu_h"

class L1SpikeBuffer
{
	public :

		// default ctor is fine here

		void allocate(size_t partitionCount);

		l1spike_t* buffer() const { return m_buffer.get(); }

		uint* heads() const { return m_heads.get(); }

	private :

		boost::shared_ptr<l1spike_t> m_buffer;
		boost::shared_ptr<uint> m_heads;
};

#endif
