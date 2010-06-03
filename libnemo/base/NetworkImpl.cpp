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



void
NetworkImpl::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char plastic)
{
	m_fcm[source][delay].push_back(synapse_t(target, weight, plastic));

	//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
	m_maxIdx = std::max(m_maxIdx, int(std::max(source, target)));
	m_minIdx = std::min(m_minIdx, int(std::min(source, target)));
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);
}



void
NetworkImpl::addSynapses(
		unsigned source,
		const std::vector<unsigned>& targets,
		const std::vector<unsigned>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char>& plastic)
{
	size_t length = targets.size();

	if(length != delays.size() || length != weights.size() || length != plastic.size()) {
		std::ostringstream msg;
		msg << "The input vectors to addSynapses (for neuron " << source << ") have different lengths\n"
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
		addSynapse(source, targets[i], delays[i], weights[i], plastic[i]);
	}
}


template
void
NetworkImpl::addSynapses<unsigned, unsigned, float, unsigned char>(unsigned,
		const unsigned[], const unsigned[], const float[],
		const unsigned char[], size_t);




template<typename N, typename D, typename W, typename B>
void
NetworkImpl::addSynapses(
		N source,
		const N targets[],
		const D delays[],
		const W weights[],
		const B plastic[],
		size_t length)
{
	using namespace boost;

	if(length == 0) {
		return;
	}

	for(size_t i=0; i < length; ++i) {
		addSynapse(
				numeric_cast<unsigned, N>(source),
				numeric_cast<unsigned, N>(targets[i]),
				numeric_cast<unsigned, D>(delays[i]),
				numeric_cast<float, W>(weights[i]),
				numeric_cast<unsigned char, B>(plastic[i]));
	}
}



unsigned
NetworkImpl::neuronCount() const
{
	return m_neurons.size();
}

}
