/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NetworkImpl.hpp"

#include <sstream>
#include <limits>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/format.hpp>

#include <nemo/bitops.h>
#include <nemo/network/programmatic/neuron_iterator.hpp>
#include <nemo/network/programmatic/synapse_iterator.hpp>
#include "exception.hpp"
#include "synapse_indices.hpp"

namespace nemo {
	namespace network {


NetworkImpl::NetworkImpl() :
	m_minIdx(std::numeric_limits<int>::max()),
	m_maxIdx(std::numeric_limits<int>::min()),
	m_maxDelay(0),
	m_minWeight(0),
	m_maxWeight(0)
{
	//! remove hard-coding of izhikevich model
	registerNeuronType(NeuronType(5,2));
}



void
NetworkImpl::registerNeuronType(const NeuronType& type)
{
	/* The current implementation does not support mixing neuron types */
	if(!m_neuronTypes.empty()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Support for multiple neuron models not yet implemented");
	}

	m_neuronTypes.push_back(type);
	mf_param.resize(type.f_nParam());
	mf_state.resize(type.f_nState());
}



void
NetworkImpl::addNeuron(unsigned nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	using boost::format;
	if(m_mapper.find(nidx) != m_mapper.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Duplicate neuron index for neuron %u") % nidx));
	}
	m_maxIdx = std::max(m_maxIdx, int(nidx));
	m_minIdx = std::min(m_minIdx, int(nidx));

	m_mapper[nidx] = mf_param.at(0).size();
	mf_param.at(0).push_back(a);
	mf_param.at(1).push_back(b);
	mf_param.at(2).push_back(c);
	mf_param.at(3).push_back(d);
	mf_param.at(4).push_back(sigma);
	mf_state.at(0).push_back(u);
	mf_state.at(1).push_back(v);
}



void
NetworkImpl::setNeuron(unsigned nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	size_t i = existingNeuronLocation(nidx);
	mf_param.at(0).at(i) = a;
	mf_param.at(1).at(i) = b;
	mf_param.at(2).at(i) = c;
	mf_param.at(3).at(i) = d;
	mf_param.at(4).at(i) = sigma;
	mf_state.at(0).at(i) = u;
	mf_state.at(1).at(i) = v;
}



synapse_id
NetworkImpl::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char plastic)
{
	using boost::format;

	if(delay < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid delay (%u) for synapse between %u and %u") % delay % source % target));
	}

	id32_t id = m_fcm[source].addSynapse(target, delay, weight, plastic != 0);

	//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
	m_maxIdx = std::max(m_maxIdx, int(std::max(source, target)));
	m_minIdx = std::min(m_minIdx, int(std::min(source, target)));
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);

	return make_synapse_id(source, id);
}



size_t
NetworkImpl::existingNeuronLocation(unsigned nidx) const
{
	using boost::format;
	mapper_t::const_iterator found = m_mapper.find(nidx);
	if(found == m_mapper.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron index %u") % nidx));
	}
	return found->second;
}



std::deque<float>&
NetworkImpl::f_parameter(size_t i)
{
	return const_cast<std::deque<float>&>(
			static_cast<const NetworkImpl&>(*this).f_parameter(i));
}



const std::deque<float>&
NetworkImpl::f_parameter(size_t i) const
{
	using boost::format;
	if(i >= mf_param.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid parameter index %u") % i));
	}
	return mf_param[i];
}



std::deque<float>&
NetworkImpl::f_state(size_t i)
{
	return const_cast<std::deque<float>&>(
			static_cast<const NetworkImpl&>(*this).f_state(i));
}



const std::deque<float>&
NetworkImpl::f_state(size_t i) const
{
	using boost::format;
	if(i >= mf_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid state variable index %u") % i));
	}
	return mf_state[i];
}



float
NetworkImpl::getNeuronState(unsigned nidx, unsigned var) const
{
	return f_state(var).at(existingNeuronLocation(nidx));
}



float
NetworkImpl::getNeuronParameter(unsigned nidx, unsigned parameter) const
{
	return f_parameter(parameter).at(existingNeuronLocation(nidx));
}



void
NetworkImpl::setNeuronState(unsigned nidx, unsigned var, float val)
{
	f_state(var).at(existingNeuronLocation(nidx)) = val;
}



void
NetworkImpl::setNeuronParameter(unsigned nidx, unsigned parameter, float val)
{
	f_parameter(parameter).at(existingNeuronLocation(nidx)) = val;
}



const Axon&
NetworkImpl::axon(nidx_t source) const
{
	using boost::format;
	fcm_t::const_iterator i_src = m_fcm.find(source);
	if(i_src == m_fcm.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("synapses of non-existing neuron (%u) requested") % source));
	}
	return i_src->second;
}



unsigned
NetworkImpl::getSynapseTarget(const synapse_id& id) const
{
	return axon(neuronIndex(id)).getTarget(synapseIndex(id));
}



unsigned
NetworkImpl::getSynapseDelay(const synapse_id& id) const
{
	return axon(neuronIndex(id)).getDelay(synapseIndex(id));
}



float
NetworkImpl::getSynapseWeight(const synapse_id& id) const
{
	return axon(neuronIndex(id)).getWeight(synapseIndex(id));
}



unsigned char
NetworkImpl::getSynapsePlastic(const synapse_id& id) const
{
	return axon(neuronIndex(id)).getPlastic(synapseIndex(id));
}



const std::vector<synapse_id>&
NetworkImpl::getSynapsesFrom(unsigned source)
{
	fcm_t::const_iterator i_src = m_fcm.find(source);
	if(i_src == m_fcm.end()) {
		m_queriedSynapseIds.clear();
	} else {
		i_src->second.setSynapseIds(source, m_queriedSynapseIds);
	}
	return m_queriedSynapseIds;
}



unsigned
NetworkImpl::neuronCount() const
{
	return mf_param.empty() ? 0 : mf_param.at(0).size();
}


nidx_t
NetworkImpl::minNeuronIndex() const
{
	if(neuronCount() == 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"minimum neuron index requested for empty network");
	}
	return m_minIdx;
}


nidx_t
NetworkImpl::maxNeuronIndex() const
{
	if(neuronCount() == 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				"maximum neuron index requested for empty network");
	}
	return m_maxIdx;
}


/* Neuron iterators */



neuron_iterator
NetworkImpl::neuron_begin() const
{
	return neuron_iterator(new programmatic::neuron_iterator(m_mapper.begin(), mf_param, mf_state, m_neuronTypes.at(0)));
}


neuron_iterator
NetworkImpl::neuron_end() const
{
	return neuron_iterator(new programmatic::neuron_iterator(m_mapper.end(), mf_param, mf_state, m_neuronTypes.at(0)));
}


synapse_iterator
NetworkImpl::synapse_begin() const
{
	fcm_t::const_iterator ni = m_fcm.begin();
	fcm_t::const_iterator ni_end = m_fcm.end();

	size_t gi = 0;
	size_t gi_end = 0;

	if(ni != ni_end) {
		gi_end = ni->second.size();
	}
	return synapse_iterator(
		new programmatic::synapse_iterator(ni, ni_end, gi, gi_end));
}


synapse_iterator
NetworkImpl::synapse_end() const
{
	fcm_t::const_iterator ni = m_fcm.end();
	size_t gi = 0;

	if(m_fcm.begin() != ni) {
		gi = m_fcm.rbegin()->second.size();
	}

	return synapse_iterator(new programmatic::synapse_iterator(ni, ni, gi, gi));
}


}	}
