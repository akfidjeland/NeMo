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
	;
}



unsigned
NetworkImpl::addNeuronType(const NeuronType& type)
{
	if(!m_neurons.empty()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Only a single neuron type per network is supported");
	}
	unsigned type_id = m_neurons.size();
	m_neurons.push_back(Neurons(type));
	return type_id;
}



const Neurons&
NetworkImpl::neuronCollection(unsigned type_id) const
{
	using boost::format;
	if(type_id >= m_neurons.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron type id %u") % type_id));
	}
	return m_neurons[type_id];
}



Neurons&
NetworkImpl::neuronCollection(unsigned type_id)
{
	return const_cast<Neurons&>(static_cast<const NetworkImpl&>(*this).neuronCollection(type_id));
}



void
NetworkImpl::addNeuron(
		unsigned type_id,
		unsigned nidx,
		const float param[],
		const float state[])
{
	using boost::format;
	if(m_mapper.find(nidx) != m_mapper.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Duplicate neuron index for neuron %u") % nidx));
	}
	m_maxIdx = std::max(m_maxIdx, int(nidx));
	m_minIdx = std::min(m_minIdx, int(nidx));

	unsigned l_nidx = neuronCollection(type_id).add(param, state);
	m_mapper[nidx] = NeuronAddress(type_id, l_nidx);
}




void
NetworkImpl::setNeuron(unsigned nidx, const float param[], const float state[])
{
	const NeuronAddress& addr = existingNeuronAddress(nidx);
	neuronCollection(addr.first).set(addr.second, param, state);
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



const NetworkImpl::NeuronAddress&
NetworkImpl::existingNeuronAddress(unsigned nidx) const
{
	using boost::format;
	mapper_t::const_iterator found = m_mapper.find(nidx);
	if(found == m_mapper.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron index %u") % nidx));
	}
	return found->second;
}



float
NetworkImpl::getNeuronState(unsigned nidx, unsigned var) const
{
	const NeuronAddress& addr = existingNeuronAddress(nidx);
	return neuronCollection(addr.first).getState(addr.second, var);
}



float
NetworkImpl::getNeuronParameter(unsigned nidx, unsigned parameter) const
{
	const NeuronAddress& addr = existingNeuronAddress(nidx);
	return neuronCollection(addr.first).getParameter(addr.second, parameter);
}



void
NetworkImpl::setNeuronState(unsigned nidx, unsigned var, float val)
{
	const NeuronAddress& addr = existingNeuronAddress(nidx);
	return neuronCollection(addr.first).setState(addr.second, var, val);
}



void
NetworkImpl::setNeuronParameter(unsigned nidx, unsigned parameter, float val)
{
	const NeuronAddress& addr = existingNeuronAddress(nidx);
	return neuronCollection(addr.first).setParameter(addr.second, parameter, val);
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
	unsigned total = 0;
	for(std::vector<Neurons>::const_iterator i = m_neurons.begin();
			i != m_neurons.end(); ++i) {
		total += i->size();
	}
	return total;
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
	/* only a single neuron model supported here */
	if(m_neurons.size() != 1) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "No neurons in network");
	}
	const Neurons& neurons = m_neurons.front();
	return neuron_iterator(new programmatic::neuron_iterator(m_mapper.begin(),
				neurons.mf_param, neurons.mf_state, neurons.type()));
}


neuron_iterator
NetworkImpl::neuron_end() const
{
	/* only a single neuron model supported here */
	if(m_neurons.size() != 1) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "No neurons in network");
	}
	const Neurons& neurons = m_neurons.front();
	return neuron_iterator(new programmatic::neuron_iterator(m_mapper.end(),
				neurons.mf_param, neurons.mf_state, neurons.type()));
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
