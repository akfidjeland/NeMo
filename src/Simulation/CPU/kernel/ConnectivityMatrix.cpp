#include "ConnectivityMatrix.hpp"

#include <assert.h>
#include <stdlib.h>

#include <stdexcept>
#include <algorithm>

#include "common.h"


namespace nemo {
	namespace cpu {

bool
operator<(const ForwardIdx& a, const ForwardIdx& b)
{
	return a.source < b.source || (a.source == b.source && a.delay < b.delay);
}


ConnectivityMatrix::ConnectivityMatrix():
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
		const nidx_t targets[],
		const weight_t weights[],
		const uint plastic[],
		size_t len)
{
	if(delay <= 0) {
		throw std::domain_error("zero or negative delay in connectivity matrix construction");
	}

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
		if(plastic[i]) {
			Incoming& inc = m_racc[targets[i]];
			inc.addresses.push_back(RSynapse(source, delay));
			inc.w_diff.push_back(0.0);
		}
	}

	m_maxDelay = std::max(m_maxDelay, delay);
	m_sourceIndices.insert(source);
}



void
ConnectivityMatrix::finalize()
{
	if(!m_finalized && !m_sourceIndices.empty()) {

		nidx_t maxIdx = *(std::max_element(m_sourceIndices.begin(), m_sourceIndices.end()));
		m_cm.resize((maxIdx+1) * m_maxDelay);

		for(nidx_t n=0; n <= maxIdx; ++n) {
			for(delay_t d=1; d <= m_maxDelay; ++d) {
				std::map<ForwardIdx, Row>::const_iterator row = m_acc.find(ForwardIdx(n, d));
				if(row != m_acc.end()) {
					m_cm.at(addressOf(n,d)) = row->second;
				} else {
					m_cm.at(addressOf(n,d)) = Row(); // defaults to empty row
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
	return m_cm.at(addressOf(source, delay));
}



Incoming&
ConnectivityMatrix::getIncoming(nidx_t target)
{
	assert(m_finalized);
	return m_racc.find(target)->second;
}


	} // namespace cpu
} // namespace nemo
