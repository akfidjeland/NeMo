#include "FiringBuffer.hpp"
#include "exception.hpp"

namespace nemo {


FiringBuffer::FiringBuffer() :
	m_oldestCycle(-1)
{
	/* Need a dummy entry, to pop on first call to readFiring */
	m_fired.push_back(std::vector<unsigned>());
}


std::vector<unsigned>&
FiringBuffer::enqueue()
{
	m_oldestCycle += 1;
	m_fired.push_back(std::vector<unsigned>());
	return m_fired.back();
}



FiredList
FiringBuffer::dequeue()
{
	if(m_fired.size() == 0) {
		throw nemo::exception(NEMO_BUFFER_UNDERFLOW, "Firing buffer underflow");
	}
	m_fired.pop_front();
	return FiredList(m_oldestCycle, m_fired.front());
}


}
