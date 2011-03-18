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


Row::Row(const std::vector<FAxonTerminal>& ss) :
	len(ss.size())
{
	FAxonTerminal* ptr;
#ifdef HAVE_POSIX_MEMALIGN
	//! \todo factor out the memory aligned allocation
	int error = posix_memalign((void**)&ptr,
			ASSUMED_CACHE_LINE_SIZE,
			ss.size()*sizeof(FAxonTerminal));
	if(error) {
		throw nemo::exception(NEMO_ALLOCATION_ERROR, "Failed to allocate CM row");
	}
#else
	ptr = (FAxonTerminal*) malloc(ss.size()*sizeof(FAxonTerminal));
#endif

	std::copy(ss.begin(), ss.end(), ptr);
	data = boost::shared_array<FAxonTerminal>(ptr, free);
}




ConnectivityMatrix::ConnectivityMatrix(
		const ConfigurationImpl& conf,
		const mapper_t& mapper) :
	m_mapper(mapper),
	m_fractionalBits(conf.fractionalBits()),
	m_maxDelay(0),
	m_writeOnlySynapses(conf.writeOnlySynapses())
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
	m_maxDelay(0),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}

	network::synapse_iterator i = net.synapse_begin();
	network::synapse_iterator i_end = net.synapse_end();

	for( ; i != i_end; ++i) {
		addSynapse(mapper.localIdx(i->source), mapper.localIdx(i->target()), *i);
	}
}



void
ConnectivityMatrix::addSynapse(nidx_t source, nidx_t target, const Synapse& s)
{
	delay_t delay = s.delay;
	fix_t weight = fx_toFix(s.weight(), m_fractionalBits);

	fidx_t fidx(source, delay);
	row_t& row = m_acc[fidx];
	sidx_t sidx = row.size();
	row.push_back(FAxonTerminal(target, weight));

	//! \todo could do this on finalize pass, since there are fewer steps there
	m_delaysAcc.addDelay(source, delay);

	unsigned char plastic = s.plastic();
	if(plastic) {
		m_racc[target].push_back(RSynapse(source, delay, sidx));
	}

	if(!m_writeOnlySynapses) {
		/* The auxillary synapse maps always uses the global (user-specified)
		 * source and target neuron ids, since it's used for lookups basd on
		 * these global ids */
		aux_row& auxRow = m_cmAux[s.source];
		insert(s.id(), AxonTerminalAux(sidx, delay, plastic != 0), auxRow);
	}
}



void
ConnectivityMatrix::finalize(const mapper_t& mapper, bool verifySources)
{
	finalizeForward(mapper, verifySources);
}



/* The fast lookup is indexed by source and delay. */
void
ConnectivityMatrix::finalizeForward(const mapper_t& mapper, bool verifySources)
{
	m_maxDelay = m_delaysAcc.maxDelay();
	m_delays.init(m_delaysAcc);
	m_delaysAcc.clear();

	if(m_acc.empty()) {
		return;
	}

	/* This relies on lexicographical ordering of tuple */
	nidx_t maxSourceIdx = m_acc.rbegin()->first.get<0>();
	m_cm.resize((maxSourceIdx+1) * m_maxDelay);

	//! \todo change order here: default to Row() in all location, and then just iterate over map
	for(nidx_t n=0; n <= maxSourceIdx; ++n) {
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
				verifySynapseTerminals(row->first, row->second, mapper, verifySources);
				m_cm.at(addressOf(n,d)) = Row(row->second);
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
		const mapper_t& mapper,
		bool verifySource) const
{
	using boost::format;

	if(verifySource) {
		nidx_t source = idx.get<0>();
		if(!mapper.existingLocal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid synapse source neuron %u") % source));
		}
	}

	for(size_t s=0; s < row.size(); ++s) {
		nidx_t target = row.at(s).target;
		if(!mapper.existingLocal(target)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse target neuron %u (source: %u)") % target % idx.get<0>()));
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
	if(!m_stdp) {
		throw exception(NEMO_LOGIC_ERROR, "applyStdp called, but no STDP model specified");
	}

	fix_t fx_reward = fx_toFix(reward, m_fractionalBits);

	for(std::map<nidx_t, Incoming>::iterator row = m_racc.begin(); row != m_racc.end(); ++row) {

		Incoming& incoming = row->second;

		for(Incoming::iterator s = incoming.begin(); s != incoming.end(); ++s) {


			if(fx_reward != 0U) {
				fix_t* w_old = weight(*s);
				fix_t w_new = m_stdp->updatedWeight(*w_old, fx_mul(fx_reward, s->w_diff, m_fractionalBits));

				if(*w_old != w_new) {
#ifdef DEBUG_TRACE
					fprintf(stderr, "stdp (%u -> %u) %f %+f = %f\n",
							s->source, row->first, fx_toFloat(*w_old, m_fractionalBits),
							fx_toFloat(fx_mul(reward, s->w_diff, m_fractionalBits), m_fractionalBits),
							fx_toFloat(w_new, m_fractionalBits));
#endif
					*w_old = w_new;
				}
			}
			s->w_diff = 0;
		}
	}
}



const std::vector<synapse_id>&
ConnectivityMatrix::getSynapsesFrom(unsigned source)
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	/* The relevant data is stored in the auxillary synapse map, which is
	 * already indexed in global neuron indices. Therefore, no need to map into
	 * local ids */
	size_t nSynapses = 0;
	aux_map::const_iterator iRow = m_cmAux.find(source);
	if(iRow == m_cmAux.end()) {
		if(!m_mapper.existingGlobal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing source neuron id (%u) in synapse id query") % source));
		}
		/* else just leave nSynapses at zero */
	} else {
		/* Synapse ids are consecutive */
		nSynapses = iRow->second.size();
	}

	m_queriedSynapseIds.resize(nSynapses);

	for(size_t iSynapse = 0; iSynapse < nSynapses; ++iSynapse) {
		m_queriedSynapseIds[iSynapse] = make_synapse_id(source, iSynapse);
	}

	return m_queriedSynapseIds;
}



const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	return m_cm.at(addressOf(source, delay));
}



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(const synapse_id& id) const
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	nidx_t neuron = neuronIndex(id);
	aux_map::const_iterator it = m_cmAux.find(neuron);
	if(it == m_cmAux.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron id (%u) in synapse query") % neuron));
	}
	return it->second.at(synapseIndex(id));
}



unsigned
ConnectivityMatrix::getTarget(const synapse_id& id) const
{
	const AxonTerminalAux& s = axonTerminalAux(id);
	nidx_t l_source = m_mapper.localIdx(neuronIndex(id));
	nidx_t l_target = m_cm[addressOf(l_source, s.delay)].data[s.idx].target;
	return m_mapper.globalIdx(l_target);
}



float
ConnectivityMatrix::getWeight(const synapse_id& id) const
{
	const AxonTerminalAux& s = axonTerminalAux(id);
	nidx_t l_source = m_mapper.localIdx(neuronIndex(id));
	const Row& row = m_cm[addressOf(l_source, s.delay)];
	assert(s.idx < row.len);
	fix_t w = row.data[s.idx].weight;
	return fx_toFloat(w, m_fractionalBits);
}



unsigned
ConnectivityMatrix::getDelay(const synapse_id& id) const
{
	return axonTerminalAux(id).delay;
}



unsigned char
ConnectivityMatrix::getPlastic(const synapse_id& id) const
{
	return axonTerminalAux(id).plastic;
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_begin(nidx_t source) const
{
	return m_delays.begin(source);
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_end(nidx_t source) const
{
	return m_delays.end(source);
}


} // namespace nemo
