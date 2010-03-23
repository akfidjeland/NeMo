#include "SynapseAddressTable.hpp"

#include <stdexcept>

namespace nemo {


void
SynapseAddressTable::addBlock(nidx_t sourceNeuron, uint blockStart, uint blockEnd)
{
	m_data[sourceNeuron].second.push_back(AddressRange(blockStart, blockEnd));
}



void
SynapseAddressTable::setWarpRange(nidx_t sourceNeuron, uint start, uint end)
{
	m_data[sourceNeuron].first = AddressRange(start, end);
}



const AddressRange&
SynapseAddressTable::warpsOf(nidx_t sourceNeuron) const
{
	std::map<nidx_t, neuron_data_t>::const_iterator v = m_data.find(sourceNeuron);
	if(v == m_data.end()) {
		throw std::out_of_range("Invalid source neuron");
	}
	const AddressRange& range = v->second.first;
	if(!range.valid()) {
		throw std::logic_error("Incomplete warp range requested");
	}
	return range;
}



const std::vector<AddressRange>&
SynapseAddressTable::synapsesOf(nidx_t sourceNeuron) const
{
	std::map<nidx_t, neuron_data_t>::const_iterator v = m_data.find(sourceNeuron);
	if(v == m_data.end()) {
		throw std::out_of_range("Invalid source neuron");
	}
	return v->second.second;
}


} // end namespace nemo
