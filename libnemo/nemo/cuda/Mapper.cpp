#include "Mapper.hpp"

#include <cassert>
#include <boost/format.hpp>

#include <nemo/exception.hpp>
#include <nemo/util.h>

#include "kernel.cu_h"


namespace nemo {
	namespace cuda {

using boost::format;

Mapper::Mapper(const nemo::NetworkImpl& net, unsigned partitionSize) :
	m_partitionSize(partitionSize),
	m_partitionCount(0),
	m_offset(0)
{
	/* The partition size validity will already have been tested. */
	assert(m_partitionSize <= MAX_PARTITION_SIZE && m_partitionSize >= THREADS_PER_BLOCK);

	if(net.neuronCount() > 0) {
		unsigned ncount = net.maxNeuronIndex() - net.minNeuronIndex() + 1;
		m_partitionCount = DIV_CEIL(ncount, m_partitionSize);
		m_offset = net.minNeuronIndex();
	}
}


DeviceIdx
Mapper::deviceIdx(nidx_t global) const
{
	nidx_t local = global - m_offset;
	assert(global >= m_offset);
	return DeviceIdx(local / m_partitionSize, local % m_partitionSize);
}



unsigned
Mapper::maxHostIdx() const
{
	return m_offset + m_partitionCount * m_partitionSize - 1;
}


}	}
