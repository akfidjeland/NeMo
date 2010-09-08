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
#include "exception.hpp"

namespace nemo {


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
	m_fcm[source][delay].push_back(synapse_t(count, target, weight, plastic));
	count += 1;

	//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
	m_maxIdx = std::max(m_maxIdx, int(std::max(source, target)));
	m_minIdx = std::min(m_minIdx, int(std::min(source, target)));
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);

	//! \todo do a range check here
	return (id64_t(source) << 32) | id64_t(id);
}



void
NetworkImpl::addSynapses(
		const std::vector<unsigned>& sources,
		const std::vector<unsigned>& targets,
		const std::vector<unsigned>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char>& plastic)
{
	size_t length = sources.size();

	if(length != targets.size() 
			|| length != delays.size() 
			|| length != weights.size() 
			|| length != plastic.size()) {
		std::ostringstream msg;
		msg << "Input vectors to addSynapses have different lengths\n"
			<< "\tsources: " << sources.size() << std::endl
			<< "\ttargets: " << targets.size() << std::endl
			<< "\tdelays: " << delays.size() << std::endl
			<< "\tweights: " << weights.size() << std::endl
			<< "\tplastic: " << plastic.size() << std::endl;
		throw nemo::exception(NEMO_INVALID_INPUT, msg.str());
	}

    if(length == 0) {
        return;
	}

	for(size_t i=0; i < length; ++i) {
		addSynapse(sources[i], targets[i], delays[i], weights[i], plastic[i]);
	}
}


template
void
NetworkImpl::addSynapses<unsigned, unsigned, float, unsigned char>(
		const unsigned[], const unsigned[], const unsigned[], const float[],
		const unsigned char[], size_t);




template<typename N, typename D, typename W, typename B>
void
NetworkImpl::addSynapses(
		const N sources[],
		const N targets[],
		const D delays[],
		const W weights[],
		const B plastic[],
		size_t length)
{
	using namespace boost;

	//! \todo do a length check here as well

	if(length == 0) {
		return;
	}

	for(size_t i=0; i < length; ++i) {
		addSynapse(
				numeric_cast<unsigned, N>(sources[i]),
				numeric_cast<unsigned, N>(targets[i]),
				numeric_cast<unsigned, D>(delays[i]),
				numeric_cast<float, W>(weights[i]),
				numeric_cast<unsigned char, B>(plastic[i]));
	}
}



void
NetworkImpl::getSynapses(
		unsigned source,
		std::vector<unsigned>& targets,
		std::vector<unsigned>& delays,
		std::vector<float>& weights,
		std::vector<unsigned char>& plastic) const
{
	using boost::format;

	fcm_t::const_iterator i_src = m_fcm.find(source);
	if(i_src == m_fcm.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("synapses of non-existing neuron (%u) requested") % source));
	}

	targets.clear();
	delays.clear();
	weights.clear();
	plastic.clear();

	const axon_t& axon = i_src->second;
	for(axon_t::const_iterator i_axon = axon.begin(); i_axon != axon.end(); ++i_axon) {
		unsigned delay = i_axon->first;
		const bundle_t& bundle = i_axon->second;
		for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
			targets.push_back(s->target);
			delays.push_back(delay);
			weights.push_back(s->weight);
			plastic.push_back(s->plastic);
		}
	}
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


}
