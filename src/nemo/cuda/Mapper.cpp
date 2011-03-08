#include "Mapper.hpp"

#include <cassert>
#include <boost/format.hpp>

#include <nemo/network/Generator.hpp>
#include <nemo/exception.hpp>
#include <nemo/util.h>

#include "kernel.cu_h"


namespace nemo {
	namespace cuda {


Mapper::Mapper(const nemo::network::Generator& net, unsigned partitionSize) :
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
Mapper::addIdx(nidx_t global)
{
	m_existingGlobal.insert(global);
	return deviceIdx(global);
}



nidx_t
Mapper::localIdx(const DeviceIdx& d) const
{
	nidx_t local = d.partition * m_partitionSize + d.neuron;
#ifndef NDEBUG
	using boost::format;
	if(!existingLocal(local)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u (p%un%u) refers to a non-existing neuron")
					% globalIdx(local) % d.partition % d.neuron));
	}
#endif
	return local;
}


nidx_t
Mapper::globalIdx(pidx_t p, nidx_t n) const
{
	nidx_t global = m_offset + p * m_partitionSize + n;
#ifndef NDEBUG
	using boost::format;
	if(!existingGlobal(global)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u (p%un%u) refers to a non-existing neuron") % global % p % n));
	}
#endif
	return global;
}



DeviceIdx
Mapper::deviceIdx(nidx_t global) const
{
	nidx_t local = global - m_offset;
	assert(global >= m_offset);
	return DeviceIdx(local / m_partitionSize, local % m_partitionSize);
}



DeviceIdx
Mapper::existingDeviceIdx(nidx_t global) const
{
	using boost::format;
	if(!existingGlobal(global)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron index %u refers to non-existing neuron") % global));
	}
	return deviceIdx(global);
}



unsigned
Mapper::maxHandledGlobalIdx() const
{
	return m_offset + m_partitionCount * m_partitionSize - 1;
}


}	}