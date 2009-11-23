#include "ConnectivityMatrix.hpp"

#include <assert.h>
#include <stdlib.h>

#include "common.h"


bool
operator<(const ForwardIdx& a, const ForwardIdx& b)
{
	return a.source < b.source || (a.source == b.source && a.delay < b.delay);
}


ConnectivityMatrix::ConnectivityMatrix(size_t neuronCount):
	m_neuronCount(neuronCount),
	m_maxDelay(0),
	m_finalized(false)
{ }



ConnectivityMatrix::~ConnectivityMatrix()
{
	/* The 'Row' struct does not have its own destructor, to keep things a bit
	 * simpler, so need to clean up here. */
	for(std::map<ForwardIdx, Row>::const_iterator i = m_acc.begin();
			i != m_acc.end(); ++i) {
		Synapse* data = i->second.data;
		if(data != NULL) {
			free(data);
		}
	}
}




void
ConnectivityMatrix::setRow(
		nidx_t source,
		delay_t delay,
		const nidx_t* targets,
		const weight_t* weights,
		size_t len)
{
	assert(delay > 0);

	Row& ss = m_acc[ForwardIdx(source, delay)];

	/* It's not clear whether alligning this data to cache lines have any
	 * effect on performance, but it can't hurt either. */
	//! \todo only do alligned allocation if posix_memalign is available
	int error = posix_memalign((void**)&ss.data,
			ASSUMED_CACHE_LINE_SIZE,
			len*sizeof(Synapse)); \
	//! \todo deal with allocation errors
	ss.len = len;

	for(size_t i=0; i<len; ++i) {
		ss.data[i] = Synapse(weights[i], targets[i]);
	}

	m_maxDelay = std::max(m_maxDelay, delay);
}



void
ConnectivityMatrix::finalize()
{
	if(!m_finalized) {
		m_cm.resize(m_neuronCount * m_maxDelay);

		for(int n=0; n < m_neuronCount; ++n) {
			for(int d=1; d <= m_maxDelay; ++d) {
				std::map<ForwardIdx, Row>::const_iterator row = m_acc.find(ForwardIdx(n, d));
				if(row != m_acc.end()) {
					m_cm[addressOf(n,d)] = row->second;
				} else {
					m_cm[addressOf(n,d)] = Row(); // defaults to empty row
				}
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
