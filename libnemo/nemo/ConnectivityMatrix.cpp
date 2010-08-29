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
#include <nemo/network/Generator.hpp>
#include "ConfigurationImpl.hpp"
#include "exception.hpp"
#include "fixedpoint.hpp"
#include "synapse_indices.hpp"


namespace nemo {


Row::Row(const std::vector<AxonTerminal>& ss, unsigned fbits) :
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
	for(std::vector<nemo::AxonTerminal>::const_iterator si = ss.begin();
			si != ss.end(); ++si) {
		size_t i = si - ss.begin();
		ptr[i] = FAxonTerminal<fix_t>(fx_toFix(si->weight, fbits), si->target);
	}
}




ConnectivityMatrix::ConnectivityMatrix(
		const ConfigurationImpl& conf,
		const mapper_t& mapper) :
	m_mapper(mapper),
	m_fractionalBits(conf.fractionalBits()),
	m_maxDelay(0)
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}
}


/* Insert into vector, resizing if appropriate */
template<typename T>
void
insert(size_t idx, const T& val, std::vector<T>& vec)
{
	if(idx >= vec.size()) {
		vec.resize(idx+1);
	}
	vec.at(idx) = val;
}



ConnectivityMatrix::ConnectivityMatrix(
		const network::Generator& net,
		const ConfigurationImpl& conf,
		const mapper_t& mapper) :
	m_mapper(mapper),
	m_fractionalBits(conf.fractionalBits()),
	m_maxDelay(0)
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}

	network::synapse_iterator i = net.synapse_begin();
	network::synapse_iterator i_end = net.synapse_end();

	for( ; i != i_end; ++i) {

		nidx_t source = mapper.localIdx(i->source);
		nidx_t target = mapper.localIdx(i->target());
		delay_t delay = i->delay;
		unsigned char plastic = i->plastic();

		//! \todo could also do fixed-point conversion here
		fidx_t fidx(source, delay);
		row_t& row = m_acc[fidx];
		sidx_t sidx = row.size();
		row.push_back(AxonTerminal(i->id(), target, i->weight(), plastic));

		//! \todo could do this on finalize pass, since there are fewer steps there
		m_delays[source].insert(delay);
		m_maxDelay = std::max(m_maxDelay, delay);

		if(plastic) {
			m_racc[target].push_back(RSynapse(source, delay, sidx));
		}

		aux_row& auxRow = m_cmAux[source];
		insert(i->id(), AxonTerminalAux(sidx, delay, plastic), auxRow);
	}
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

	//! \todo change order here: default to Row() in all location, and then just iterate over map
	for(nidx_t n=0; n <= maxIdx; ++n) {
		for(delay_t d=1; d <= m_maxDelay; ++d) {

#if 0
			if(d < 1) {
				//! \todo make sure to report global index again here
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Neuron %u has synapses with delay < 1 (%u)") % source % delay));
			}
#endif

			std::map<fidx_t, row_t>::const_iterator row = m_acc.find(fidx_t(n, d));
			if(row != m_acc.end()) {
				verifySynapseTerminals(row->first, row->second, mapper);
				//! \todo make input a row in the correct format.
				m_cm.at(addressOf(n,d)) = Row(row->second, m_fractionalBits);
			} else {
				/* Insertion into map does not invalidate existing iterators */
				m_cm.at(addressOf(n,d)) = Row(); // defaults to empty row
			}
			//! \todo can delete the map now
		}
	}
}



void
ConnectivityMatrix::verifySynapseTerminals(fidx_t idx,
		const row_t& row,
		const mapper_t& mapper) const
{
	using boost::format;

	nidx_t source = idx.get<0>();
	if(!mapper.validLocal(source)) {
		throw nemo::exception(NEMO_INVALID_INPUT,
			str(format("Invalid synapse source neuron %u") % source));
	}

	for(size_t s=0; s < row.size(); ++s) {
		nidx_t target = row.at(s).target;
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



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(nidx_t neuron, id32_t synapse) const
{
	using boost::format;

	aux_map::const_iterator it = m_cmAux.find(neuron);
	if(it == m_cmAux.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron id (%u) in synapse query") % neuron));
	}
	return it->second.at(synapse);
}



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(const synapse_id& id) const
{
	return  axonTerminalAux(m_mapper.localIdx(neuronIndex(id)), synapseIndex(id));
}



const std::vector<unsigned>&
ConnectivityMatrix::getTargets(const std::vector<synapse_id>& synapses)
{
	m_queriedTargets.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses[i];
		nidx_t l_source = m_mapper.localIdx(neuronIndex(id));
		const AxonTerminalAux& s = axonTerminalAux(l_source, synapseIndex(id));
		nidx_t l_target = m_cm[addressOf(l_source, s.delay)].data[s.idx].target;
		m_queriedTargets[i] = m_mapper.globalIdx(l_target);
	}
	return m_queriedTargets;
}



const std::vector<float>&
ConnectivityMatrix::getWeights(const std::vector<synapse_id>& synapses)
{
	m_queriedWeights.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses[i];
		nidx_t source = m_mapper.localIdx(neuronIndex(id));
		const AxonTerminalAux& s = axonTerminalAux(source, synapseIndex(id));
		const Row& row = m_cm[addressOf(source, s.delay)];
		assert(s.idx < row.len);
		fix_t w = row.data[s.idx].weight;
		m_queriedWeights[i] = fx_toFloat(w, m_fractionalBits);
	}
	return m_queriedWeights;
}



const std::vector<unsigned>&
ConnectivityMatrix::getDelays(const std::vector<synapse_id>& synapses)
{
	m_queriedDelays.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		m_queriedDelays[i] = axonTerminalAux(synapses[i]).delay;
	}
	return m_queriedDelays;
}



const std::vector<unsigned char>&
ConnectivityMatrix::getPlastic(const std::vector<synapse_id>& synapses)
{
	m_queriedPlastic.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		m_queriedPlastic[i] = axonTerminalAux(synapses[i]).plastic;
	}
	return m_queriedPlastic;
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
