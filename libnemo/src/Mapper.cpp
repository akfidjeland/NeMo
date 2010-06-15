#include "Mapper.hpp"

#include <boost/format.hpp>
#include <assert.h>

#include <exception.hpp>
#include <util.h>

#include "kernel.cu_h"


namespace nemo {
	namespace cuda {

using boost::format;

Mapper::Mapper(const nemo::NetworkImpl& net, unsigned partitionSize) :
	m_partitionSize(partitionSize),
	m_partitionCount(0),
	m_offset(0)
{
	if(partitionSize > MAX_PARTITION_SIZE || partitionSize == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT, 
				str(format("Requested partition size for cuda backend (%u) not in valid range: [%u, %u]")
						% partitionSize % THREADS_PER_BLOCK % MAX_PARTITION_SIZE));
	}

	if(net.neuronCount() > 0) {
		unsigned ncount = net.maxNeuronIndex() - net.minNeuronIndex() + 1;
		m_partitionCount = DIV_CEIL(ncount, partitionSize);
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


}	}
