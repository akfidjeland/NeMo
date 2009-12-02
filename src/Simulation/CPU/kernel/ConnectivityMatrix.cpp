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
			inc.push_back(RSynapse(source, delay, i));
		}
	}

	m_maxDelay = std::max(m_maxDelay, delay);
	m_sourceIndices.insert(source);
}



void
ConnectivityMatrix::finalize()
{
	if(!m_finalized && !m_sourceIndices.empty()) {
		finalizeForward();
		finalizeReverse();
		m_finalized = true;
	}
}


void
ConnectivityMatrix::finalizeForward()
{
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
}



void
ConnectivityMatrix::finalizeReverse()
{
	nidx_t maxIdx = m_racc.rbegin()->first;
	m_wdiff.resize(maxIdx+1, Accumulator());
	for(std::map<nidx_t, Incoming>::const_iterator i = m_racc.begin();
			i != m_racc.end(); ++i) {
		nidx_t post = i->first;
		const Incoming& incoming = i->second;
		m_wdiff[post].resize(incoming.size(), 0.0);
	}
}



//! \todo have a different way to communicate non-present data
const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	assert(m_finalized);
	return m_cm.at(addressOf(source, delay));
}



const ConnectivityMatrix::Incoming&
ConnectivityMatrix::getIncoming(nidx_t target)
{
	assert(m_finalized);
	//! \todo use linear lookup here
	//return m_rcm.at(target);
	return m_racc.find(target)->second;
}


ConnectivityMatrix::Accumulator&
ConnectivityMatrix::getWAcc(nidx_t target)
{
	assert(m_finalized);
	return m_wdiff.at(target);
}



void
ConnectivityMatrix::applyStdp(double minWeight, double maxWeight, double reward)
{
	if(reward != 0.0) {
		for(std::map<nidx_t, Incoming>::iterator i = m_racc.begin();
				i != m_racc.end(); ++i) {
			applyStdpOne(i->first, i->second, minWeight, maxWeight, reward);
		}
	} else {
		for(std::vector<Accumulator>::iterator i = m_wdiff.begin();
				i != m_wdiff.end(); ++i) {
			std::fill(i->begin(), i->end(), 0.0);
		}
	}
}



weight_t*
ConnectivityMatrix::weight(const RSynapse& r_idx)
{
	Row& row = m_cm.at(addressOf(r_idx.source, r_idx.delay));
	assert(r_idx.synapse < row.len);
	return &row.data[r_idx.synapse].weight;
}


void
ConnectivityMatrix::applyStdpOne(nidx_t target,
		Incoming& incoming,
		double minWeight,
		double maxWeight,
		double reward)
{
	Accumulator& acc = getWAcc(target);

	assert(incoming.size() == acc.size());

	for(size_t s = 0; s < incoming.size(); ++s) {

		const RSynapse& rsynapse = incoming[s];

		weight_t w_diff = acc[s];

		weight_t* w_old = weight(rsynapse);
		weight_t w_new = 0.0;
		if(*w_old > 0.0f) {
			w_new = std::min(maxWeight, std::max(*w_old + w_diff, 0.0));
		} else if(*w_old < 0.0) {
			w_new = std::min(0.0, std::max(*w_old + w_diff, minWeight));
		}

		if(*w_old != w_new) {
#ifdef DEBUG_TRACE
			fprintf(stderr, "stdp (%u -> %u) %f %+f = %f\n",
					rsynapse.source, target, *w_old, w_diff, w_new);
#endif
			*w_old = w_new;
		}

		acc[s] = 0.0;
	}
}



	} // namespace cpu
} // namespace nemo
