#include "SynapseAddressTable.hpp"

#include <stdexcept>

namespace nemo {

void
SynapseAddressTable::addBlock(nidx_t sourceNeuron, uint blockStart, uint blockEnd)
{
	m_data[sourceNeuron].push_back(std::make_pair<uint, uint>(blockStart, blockEnd));
}


const std::vector<SynapseAddressTable::range_t>&
SynapseAddressTable::synapsesOf(nidx_t sourceNeuron) const
{
	std::map<nidx_t, neuron_ranges_t>::const_iterator v = m_data.find(sourceNeuron);	
	if(v == m_data.end()) {
		throw std::out_of_range("Invalid source neuron");
	}
	return v->second;
}


} // end namespace nemo
