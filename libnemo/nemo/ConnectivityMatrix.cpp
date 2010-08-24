/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConnectivityMatrix.hpp"

#include <algorithm>
#include <utility>
#include <stdlib.h>

#include <boost/tuple/tuple_comparison.hpp>
#include <boost/format.hpp>

#include <nemo/config.h>
#include "ConfigurationImpl.hpp"
#include "NetworkImpl.hpp"
#include "exception.hpp"
#include "fixedpoint.hpp"


namespace nemo {


Row::Row(const std::vector<AxonTerminal<nidx_t, weight_t> >& ss, unsigned fbits) :
	len(ss.size())
{
	FAxonTerminal<fix_t>* ptr;
#ifdef HAVE_POSIX_MEMALIGN
	//! \todo factor out the memory aligned allocation
	int error = posix_memalign((void**)&ptr,
			ASSUMED_CACHE_LINE_SIZE,
			ss.size()*sizeof(FAxonTerminal<fix_t>));
	if(error) {
		throw nemo::exception(NEMO_ALLOCATION_ERROR, "Failed to allocate CM row");
	}
#else
	ptr = (FAxonTerminal<fix_t>*) malloc(ss.size()*sizeof(FAxonTerminal<fix_t>));
#endif

	data = boost::shared_array< FAxonTerminal<fix_t> >(ptr, free);

	/* static/plastic flag is not needed in forward matrix */
	for(std::vector<nemo::AxonTerminal<nidx_t, weight_t> >::const_iterator si = ss.begin();
			si != ss.end(); ++si) {
		size_t i = si - ss.begin();
		ptr[i] = FAxonTerminal<fix_t>(fx_toFix(si->weight, fbits), si->target);
	}
}



ConnectivityMatrix::ConnectivityMatrix(const ConfigurationImpl& conf) :
	m_fractionalBits(0),
	m_maxDelay(0)
{
	if(!conf.fractionalBitsSet()) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"If constructing runtime connectivity matrix incrementally, the fixed-point format must be specified prior to construction");
	}

	m_fractionalBits = conf.fractionalBits();
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}
}



ConnectivityMatrix::ConnectivityMatrix(
		const NetworkImpl& net,
		const ConfigurationImpl& conf,
		const mapper_t& mapper) :
	m_fractionalBits(conf.fractionalBitsSet() ? conf.fractionalBits() : net.fractionalBits()),
	m_maxDelay(0)
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}

	for(std::map<nidx_t, NetworkImpl::axon_t>::const_iterator ni = net.m_fcm.begin();
			ni != net.m_fcm.end(); ++ni) {
		nidx_t source = mapper.localIdx(ni->first);
		const NetworkImpl::axon_t& axon = ni->second;
		for(NetworkImpl::axon_t::const_iterator ai = axon.begin();
				ai != axon.end(); ++ai) {
			setRow(source, ai->first, ai->second, mapper);
		}
	}
}



Row&
ConnectivityMatrix::setRow(
		nidx_t source,
		delay_t delay,
		const std::vector<AxonTerminal<nidx_t, weight_t> >& ss,
		const mapper_t& mapper)
{
	using boost::format;

	if(delay < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u has synapses with delay < 1 (%u)") % source % delay));
	}

	fidx forwardIdx(source, delay);

	std::pair<std::map<fidx, Row>::iterator, bool> insertion =
		m_acc.insert(std::make_pair<fidx, Row>(forwardIdx, Row(ss, m_fractionalBits)));

	if(!insertion.second) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Double insertion into connectivity matrix");
	}
	m_delays[source].insert(delay);
	m_maxDelay = std::max(m_maxDelay, delay);

	Row& row = insertion.first->second;
	for(size_t s=0; s < row.len; ++s) {
		nidx_t target = mapper.localIdx(row.data[s].target);
		row.data[s].target = target;
	}

	for(unsigned sidx=0; sidx < ss.size(); ++sidx) {
		const AxonTerminal<nidx_t, weight_t>& s = ss.at(sidx);
		m_plastic[forwardIdx].push_back(s.plastic);
		if(s.plastic) {
			m_racc[mapper.localIdx(s.target)].push_back(RSynapse(source, delay, sidx));
		}
	}

	return row;
}


void
ConnectivityMatrix::finalize(const mapper_t& mapper)
{
	finalizeForward(mapper);
}



/* The fast lookup is indexed by source and delay. */
void
ConnectivityMatrix::finalizeForward(const mapper_t& mapper)
{
	if(mapper.neuronCount() == 0)
		return;

	nidx_t maxIdx = mapper.maxLocalIdx();
	m_cm.resize((maxIdx+1) * m_maxDelay);

	for(nidx_t n=0; n <= maxIdx; ++n) {
		for(delay_t d=1; d <= m_maxDelay; ++d) {
			std::map<fidx, Row>::const_iterator row = m_acc.find(fidx(n, d));
			if(row != m_acc.end()) {
				verifySynapseTerminals(row->first, row->second, mapper);
				m_cm.at(addressOf(n,d)) = row->second;
			} else {
				/* Insertion into map does not invalidate existing iterators */
				m_cm.at(addressOf(n,d)) = Row(); // defaults to empty row
			}
			//! \todo can delete the map now
		}
	}
}



void
ConnectivityMatrix::verifySynapseTerminals(fidx idx,
		const Row& row,
		const mapper_t& mapper) const
{
	using boost::format;

	nidx_t source = idx.get<0>();
	if(!mapper.validLocal(source)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
			str(format("Invalid synapse source neuron %u") % source));
	}

	for(size_t s=0; s < row.len; ++s) {
		nidx_t target = row.data[s].target;
		if(!mapper.validLocal(target)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse target neuron %u (source: %u)") % target % source));
		}
	}
}



void
ConnectivityMatrix::accumulateStdp(const std::vector<uint64_t>& recentFiring)
{
	if(!m_stdp)
		return;

	//! \todo consider walking over a compact vector instead
	//! \todo could do this in multiple threads
	for(std::map<nidx_t, Incoming>::iterator i = m_racc.begin();
			i != m_racc.end(); ++i) {

		nidx_t post = i->first;
		if(recentFiring[post] & m_stdp->postFireMask()) {

			Incoming& row = i->second;

			for(Incoming::iterator s = row.begin(); s != row.end(); ++s) {
				nidx_t pre = s->source;
				uint64_t preFiring = recentFiring[pre] >> s->delay;
				fix_t w_diff = m_stdp->weightChange(preFiring, pre, post);
				//! \todo remove conditional?
				if(w_diff != 0.0) {
					s->w_diff += w_diff;
				}
			}
		}
	}
}



fix_t*
ConnectivityMatrix::weight(const RSynapse& r_idx) const
{
	const Row& row = m_cm.at(addressOf(r_idx.source, r_idx.delay));
	assert(r_idx.synapse < row.len);
	return &row.data[r_idx.synapse].weight;
}



void
ConnectivityMatrix::applyStdp(float reward)
{
	for(std::map<nidx_t, Incoming>::iterator row = m_racc.begin(); row != m_racc.end(); ++row) {

		Incoming& incoming = row->second;

		for(Incoming::iterator s = incoming.begin(); s != incoming.end(); ++s) {

			if(reward != 0.0) {
				fix_t* w_old = weight(*s);
				fix_t w_new = m_stdp->updatedWeight(*w_old, reward * s->w_diff);

				if(*w_old != w_new) {
#ifdef DEBUG_TRACE
					fprintf(stderr, "stdp (%u -> %u) %f %+f = %f\n",
							s->source, row->first, *w_old, reward * s->w_diff, w_new);
#endif
					*w_old = w_new;
				}
			}
			s->w_diff = 0;
		}
	}
}



const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	return m_cm.at(addressOf(source, delay));
}


void
ConnectivityMatrix::getSynapses(
		unsigned source,
		std::vector<unsigned>& targets,
		std::vector<unsigned>& delays,
		std::vector<float>& weights,
		std::vector<unsigned char>& plastic) const
{
	targets.clear();
	delays.clear();
	weights.clear();
	plastic.clear();

	unsigned fbits = fractionalBits();

	for(delay_iterator d = delay_begin(source), d_end = delay_end(source);
			d != d_end; ++d) {
		const Row& ss = getRow(source, *d);
		for(unsigned i = 0; i < ss.len; ++i) {
			FAxonTerminal<fix_t> s = ss.data[i];
			targets.push_back(s.target);
			weights.push_back(fx_toFloat(s.weight, fbits));
			delays.push_back(*d);
		}

		const std::vector<unsigned char>& p_row = m_plastic.find(fidx(source, *d))->second;
		std::copy(p_row.begin(), p_row.end(), std::back_inserter(plastic));
	}

	assert(plastic.size() == targets.size());
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_begin(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_delays.find(source);
	if(found == m_delays.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.begin();
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_end(nidx_t source) const
{
	std::map<nidx_t, std::set<delay_t> >::const_iterator found = m_delays.find(source);
	if(found == m_delays.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "Invalid source neuron");
	}
	return found->second.end();
}


} // namespace nemo
