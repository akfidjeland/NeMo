#include <algorithm>
#include <cassert>

#include "OutgoingDelays.hpp"
#include "types.hpp"
#include "exception.hpp"

namespace nemo {


OutgoingDelays::OutgoingDelays() :
	m_maxDelay(0)
{
	;
}


OutgoingDelays::OutgoingDelays(const OutgoingDelaysAcc& acc) :
	m_maxDelay(0)
{
	init(acc);
}


void
OutgoingDelays::init(const OutgoingDelaysAcc& acc)
{
	m_maxDelay = acc.maxDelay();
	m_data = acc.m_delays;
}



OutgoingDelays::const_iterator
OutgoingDelays::begin(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.begin();
}



OutgoingDelays::const_iterator
OutgoingDelays::end(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.end();
}



uint64_t
OutgoingDelays::delayBits(nidx_t source) const
{
	uint64_t bits = 0;
	for(const_iterator d = begin(source), d_end = end(source); d != d_end; ++d) {
		bits = bits | (uint64_t(0x1) << uint64_t(*d - 1));
	}
	return bits;
}

}
