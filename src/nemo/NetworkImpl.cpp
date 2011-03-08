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


NetworkImpl::axon_t programmatic::synapse_iterator::s_axon;
NetworkImpl::bundle_t programmatic::synapse_iterator::s_bundle;


NetworkImpl::NetworkImpl() :
	m_minIdx(std::numeric_limits<int>::max()),
	m_maxIdx(std::numeric_limits<int>::min()),
	m_maxDelay(0),
	m_minWeight(0),
	m_maxWeight(0)
{ }


void
NetworkImpl::addNeuron(unsigned nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	addNeuron(nidx, Neuron<float>(a, b, c, d, u, v, sigma));
}



void
NetworkImpl::addNeuron(nidx_t nidx, const Neuron<float>& n)
{
	using boost::format;
	if(m_neurons.find(nidx) != m_neurons.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Duplicate neuron index for neuron %u") % nidx));
	}
	m_maxIdx = std::max(m_maxIdx, int(nidx));
	m_minIdx = std::min(m_minIdx, int(nidx));
	m_neurons[nidx] = n;
}



void
NetworkImpl::setNeuron(unsigned nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	setNeuron(nidx, Neuron<float>(a, b, c, d, u, v, sigma));
}



void
NetworkImpl::setNeuron(nidx_t nidx, const Neuron<float>& n)
{
	using boost::format;
	if(m_neurons.find(nidx) == m_neurons.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Attemt to modify non-existing neuron %u") % nidx));
	}
	m_maxIdx = std::max(m_maxIdx, int(nidx));
	m_minIdx = std::min(m_minIdx, int(nidx));
	m_neurons[nidx] = n;
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

	id32_t& count = m_synapseCount[source];
	id32_t id = count;
	m_fcm[source][delay].push_back(synapse_t(count, target, weight, plastic != 0));
	count += 1;

	//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
	m_maxIdx = std::max(m_maxIdx, int(std::max(source, target)));
	m_minIdx = std::min(m_minIdx, int(std::min(source, target)));
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);

	return make_synapse_id(source, id);
}



const NetworkImpl::neuron_t&
NetworkImpl::getNeuron(unsigned nidx) const
{
	using boost::format;

	std::map<nidx_t, neuron_t>::const_iterator i = m_neurons.find(nidx);
	if(i == m_neurons.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unknown neuron index %u") % nidx));
	}
	return i->second;
}



NetworkImpl::neuron_t&
NetworkImpl::getNeuron(unsigned idx)
{
	return const_cast<neuron_t&>(static_cast<const NetworkImpl*>(this)->getNeuron(idx));
}



float
NetworkImpl::getNeuronState(unsigned nidx, unsigned var) const
{
	using boost::format;

	const neuron_t& neuron = getNeuron(nidx);
	/*! \todo change to more generic neuron storage and remove
	 * Izhikevich-specific hardcoding */
	switch(var) {
		case 0: return neuron.u;
		case 1: return neuron.v;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid neuron state variable index (%u)") % var));
	}
}



float
NetworkImpl::getNeuronParameter(unsigned nidx, unsigned parameter) const
{
	using boost::format;

	const neuron_t& neuron = getNeuron(nidx);
	/*! \todo change to more generic neuron storage and remove
	 * Izhikevich-specific hardcoding */
	switch(parameter) {
		case 0: return neuron.a;
		case 1: return neuron.b;
		case 2: return neuron.c;
		case 3: return neuron.d;
		case 4: return neuron.sigma;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid neuron parameter index (%u)") % parameter));
	}
}



void
NetworkImpl::setNeuronState(unsigned nidx, unsigned var, float val)
{
	using boost::format;

	neuron_t& neuron = getNeuron(nidx);
	/*! \todo change to more generic neuron storage and remove
	 * Izhikevich-specific hardcoding */
	switch(var) {
		case 0: neuron.u = val; break;
		case 1: neuron.v = val; break;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid neuron state variable index (%u)") % var));
	}
}



void
NetworkImpl::setNeuronParameter(unsigned nidx, unsigned parameter, float val)
{
	using boost::format;

	neuron_t& neuron = getNeuron(nidx);
	/*! \todo change to more generic neuron storage and remove
	 * Izhikevich-specific hardcoding */
	switch(parameter) {
		case 0: neuron.a = val; break;
		case 1: neuron.b = val; break;
		case 2: neuron.c = val; break;
		case 3: neuron.d = val; break;
		case 4: neuron.sigma = val; break;
		default: throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid neuron parameter index (%u)") % parameter));
	}
}



NetworkImpl::fcm_t::const_iterator
NetworkImpl::getSourceIterator(unsigned source) const
{
	using boost::format;
	fcm_t::const_iterator i_src = m_fcm.find(source);
	if(i_src == m_fcm.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("synapses of non-existing neuron (%u) requested") % source));
	}
	return i_src;
}



/* The synapse getters could do caching etc., but this is only really used in
 * testing, so is not optimised */

const std::vector<unsigned>&
NetworkImpl::getTargets(unsigned source) const
{
	fcm_t::const_iterator i_src = getSourceIterator(source);
	m_queriedTargets.clear();
	const axon_t& axon = i_src->second;
	for(axon_t::const_iterator i_axon = axon.begin(); i_axon != axon.end(); ++i_axon) {
		const bundle_t& bundle = i_axon->second;
		for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
			m_queriedTargets.push_back(s->target);
		}
	}
	return m_queriedTargets;
}



const std::vector<unsigned>&
NetworkImpl::getDelays(unsigned source) const
{
	fcm_t::const_iterator i_src = getSourceIterator(source);
	m_queriedDelays.clear();
	const axon_t& axon = i_src->second;
	for(axon_t::const_iterator i_axon = axon.begin(); i_axon != axon.end(); ++i_axon) {
		unsigned delay = i_axon->first;
		const bundle_t& bundle = i_axon->second;
		for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
			m_queriedDelays.push_back(delay);
		}
	}
	return m_queriedDelays;

}



const std::vector<float>&
NetworkImpl::getWeights(unsigned source) const
{
	fcm_t::const_iterator i_src = getSourceIterator(source);
	m_queriedWeights.clear();
	const axon_t& axon = i_src->second;
	for(axon_t::const_iterator i_axon = axon.begin(); i_axon != axon.end(); ++i_axon) {
		const bundle_t& bundle = i_axon->second;
		for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
			m_queriedWeights.push_back(s->weight);
		}
	}
	return m_queriedWeights;
}



const std::vector<unsigned char>&
NetworkImpl::getPlastic(unsigned source) const
{
	fcm_t::const_iterator i_src = getSourceIterator(source);
	m_queriedPlastic.clear();
	const axon_t& axon = i_src->second;
	for(axon_t::const_iterator i_axon = axon.begin(); i_axon != axon.end(); ++i_axon) {
		const bundle_t& bundle = i_axon->second;
		for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
			m_queriedPlastic.push_back(s->plastic);
		}
	}
	return m_queriedPlastic;
}



unsigned
NetworkImpl::neuronCount() const
{
	return m_neurons.size();
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
	return neuron_iterator(new programmatic::neuron_iterator(m_neurons.begin()));
}


neuron_iterator
NetworkImpl::neuron_end() const
{
	return neuron_iterator(new programmatic::neuron_iterator(m_neurons.end()));
}


synapse_iterator
NetworkImpl::synapse_begin() const
{
	fcm_t::const_iterator ni = m_fcm.begin();
	fcm_t::const_iterator ni_end = m_fcm.end();

	axon_t::const_iterator bi = programmatic::synapse_iterator::defaultBi();
	axon_t::const_iterator bi_end = programmatic::synapse_iterator::defaultBi();
	bundle_t::const_iterator si = programmatic::synapse_iterator::defaultSi();
	bundle_t::const_iterator si_end = programmatic::synapse_iterator::defaultSi();

	if(ni != ni_end) {
		bi = ni->second.begin();
		bi_end = ni->second.end();
		if(bi != bi_end) {
			si = bi->second.begin();
			si_end = bi->second.end();
		}
	}
	return synapse_iterator(
		new programmatic::synapse_iterator(ni, ni_end, bi, bi_end, si, si_end));
}


synapse_iterator
NetworkImpl::synapse_end() const
{
	fcm_t::const_iterator ni = m_fcm.end();
	axon_t::const_iterator bi = programmatic::synapse_iterator::defaultBi();
	bundle_t::const_iterator si = programmatic::synapse_iterator::defaultSi();

	if(m_fcm.begin() != ni) {
		const axon_t& axon = m_fcm.rbegin()->second;
		bi = axon.end();
		if(axon.begin() != bi) {
			si = axon.rbegin()->second.end();
		}
	}

	return synapse_iterator(
			new programmatic::synapse_iterator(ni, ni, bi, bi, si, si));
}


}	}
