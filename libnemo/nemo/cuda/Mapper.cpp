#include "Mapper.hpp"

#include <boost/format.hpp>
#include <assert.h>

#include <nemo/exception.hpp>
#include <nemo/util.h>

#include "kernel.cu_h"


namespace nemo {
	namespace cuda {

using boost::format;

Mapper::Mapper(const nemo::NetworkImpl& net, unsigned partitionSize) :
	m_partitionSize(partitionSize != 0 ? partitionSize : MAX_PARTITION_SIZE),
	m_partitionCount(0),
	m_offset(0)
{
	if(m_partitionSize > MAX_PARTITION_SIZE || m_partitionSize < THREADS_PER_BLOCK) {
		throw nemo::exception(NEMO_INVALID_INPUT, 
				str(format("Requested partition size for cuda backend (%u) not in valid range: [%u, %u]")
						% m_partitionSize % THREADS_PER_BLOCK % MAX_PARTITION_SIZE));
	}

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
