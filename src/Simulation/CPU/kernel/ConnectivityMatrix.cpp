#include "ConnectivityMatrix.hpp"
#include <assert.h>


ConnectivityMatrix::ConnectivityMatrix(size_t neuronCount, size_t maxDelay):
	m_neuronCount(neuronCount),
	m_maxDelay(maxDelay),
	m_cm(neuronCount * maxDelay)
{ }



void
ConnectivityMatrix::setRow(
		nidx_t source,
		delay_t delay,
		const nidx_t* targets,
		const weight_t* weights,
		size_t len)
{
	std::vector<Synapse>& ss = m_cm[addressOf(source, delay)];

	//! \todo pre-allocate length of ss
	for(size_t i=0; i<len; ++i) {
		ss.push_back(Synapse(weights[i], targets[i]));
	}
}



const std::vector<Synapse>&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	return m_cm[addressOf(source, delay)];
}



size_t
ConnectivityMatrix::addressOf(nidx_t source, delay_t delay) const
{
	//! \todo use exceptions here instead, so we can signal back to caller
	// or better yet, just make use of safe code when looking up m_cm 
	assert(source < m_neuronCount);
	assert(delay > 0);
	assert(delay <= m_maxDelay);
	return source * m_maxDelay + delay - 1;
}
