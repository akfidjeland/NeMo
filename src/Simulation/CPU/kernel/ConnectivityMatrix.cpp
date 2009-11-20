#include "ConnectivityMatrix.hpp"
#include <assert.h>


bool
operator< (const ForwardIdx& a, const ForwardIdx& b)
{
	return a.source < b.source || (a.source == b.source && a.delay < b.delay);
}


ConnectivityMatrix::ConnectivityMatrix(size_t neuronCount):
	m_neuronCount(neuronCount),
	m_maxDelay(0),
	m_finalized(false)
{ }




void
ConnectivityMatrix::setRow(
		nidx_t source,
		delay_t delay,
		const nidx_t* targets,
		const weight_t* weights,
		size_t len)
{
	assert(delay > 0);

	std::vector<Synapse>& ss = m_acc[ForwardIdx(source, delay)];

	ss.reserve(len);
	for(size_t i=0; i<len; ++i) {
		ss.push_back(Synapse(weights[i], targets[i]));
	}

	m_maxDelay = std::max(m_maxDelay, delay);
}



Row
createRow(const std::map< ForwardIdx, std::vector<Synapse> >& acc,
		nidx_t source, delay_t delay)
{
	Row ret; // defaults to empty row
	std::map<ForwardIdx, std::vector<Synapse> >::const_iterator row =
			acc.find(ForwardIdx(source ,delay));
	if(row != acc.end()) {
		ret.len = row->second.size();
		ret.data = &(row->second)[0];
	}
	return ret;
}



void
ConnectivityMatrix::finalize()
{
	if(!m_finalized) {
		m_cm.resize(m_neuronCount * m_maxDelay);

		for(int n=0; n < m_neuronCount; ++n) {
			for(int d=1; d <= m_maxDelay; ++d) {
				m_cm[addressOf(n,d)] = createRow(m_acc, n, d);
			}
		}
		m_finalized = true;
	}
}




//! \todo have a different way to communicate non-present data
const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	assert(m_finalized);
	return m_cm[addressOf(source, delay)];
}



size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	//! \todo use exceptions here instead, so we can signal back to caller
	// or better yet, just make use of safe code when looking up m_acc
	assert(source < m_neuronCount);
	assert(delay > 0);
	assert(delay <= m_maxDelay);
	return source * m_maxDelay + delay - 1;
}
