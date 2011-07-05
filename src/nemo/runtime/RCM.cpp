#include "RCM.hpp"

namespace nemo {
	namespace runtime {


RCM::RCM(construction_t& rcm) :
	m_warps(rcm.m_warps),
	m_indegree(rcm.m_dataRowLength)
{
	/* Even if there are no connections, we still need an index if the kernel
	 * assumes there /may/ be connections present. */
	bool empty = rcm.synapseCount() == 0;

	m_data.swap(rcm.m_data);
	m_forward.swap(rcm.m_forward);
	m_weights.swap(rcm.m_weights);
	if(rcm.m_stdpEnabled && !empty) {
		m_accumulator.resize(m_data.size(), 0U);
	}
}



void
RCM::clearAccumulator()
{
	std::fill(m_accumulator.begin(), m_accumulator.end(), 0U);
}


const RSynapse*
RCM::data(size_t warp) const
{
	assert(warp*WIDTH < m_data.size());
	return &m_data[warp*WIDTH];
}



fix_t*
RCM::accumulator(size_t warp)
{
	assert(warp*WIDTH < m_accumulator.size());
	return &m_accumulator[warp*WIDTH];
}



const uint32_t*
RCM::forward(size_t warp) const
{
	assert(warp*WIDTH < m_forward.size());
	return &m_forward[warp*WIDTH];
}


	} // end namespace runtime
} // end namespace nemo

