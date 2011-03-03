#include "Mapper.hpp"

#include <boost/format.hpp>
#include <nemo/exception.hpp>

namespace nemo {
	namespace cpu {


nidx_t
Mapper::addGlobal(const nidx_t& global)
{
	using boost::format;
	m_existingGlobal.insert(global);
	if(global < m_offset || global - m_offset >= m_ncount) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron index %u is not in valid range") % global));
	}
	return localIdx(global);
}



nidx_t
Mapper::existingLocalIdx(const nidx_t& global) const
{
	using boost::format;
	if(!existingGlobal(global)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron index %u refers to non-existing neuron") % global));
	}
	return localIdx(global);
}


	}
}
