#include "SynapseAddressTable.hpp"

#include <stdexcept>
#include <sstream>

#include <nemo/exception.hpp>

namespace nemo {


void
SynapseAddressTable::addBlock(nidx_t sourceNeuron, unsigned blockStart, unsigned blockEnd)
{
	m_data[sourceNeuron].second.push_back(AddressRange(blockStart, blockEnd));
}



void
SynapseAddressTable::setWarpRange(nidx_t sourceNeuron, unsigned start, unsigned end)
{
	m_data[sourceNeuron].first = AddressRange(start, end);
}



const AddressRange&
SynapseAddressTable::warpsOf(nidx_t sourceNeuron) const
{
	std::map<nidx_t, neuron_data_t>::const_iterator v = m_data.find(sourceNeuron);
	if(v == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	const AddressRange& range = v->second.first;
	if(!range.valid()) {
		std::ostringstream msg;
		msg << "Incomplete warp range requested. warpsOf(" << sourceNeuron <<
			") = " << range.start << "-" << range.end << std::endl;
		throw nemo::exception(NEMO_LOGIC_ERROR, msg.str());
	}
	return range;
}



const std::vector<AddressRange>&
SynapseAddressTable::synapsesOf(nidx_t sourceNeuron) const
{
	std::map<nidx_t, neuron_data_t>::const_iterator v = m_data.find(sourceNeuron);
	if(v == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return v->second.second;
}


} // end namespace nemo
