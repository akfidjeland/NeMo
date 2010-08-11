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



void
NetworkImpl::getSynapses(
		unsigned source,
		std::vector<unsigned>& targets,
		std::vector<unsigned>& delays,
		std::vector<float>& weights,
		std::vector<unsigned char>& plastic) const
{
	fcm_t::const_iterator i_src = m_fcm.find(source);
	if(i_src == m_fcm.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "synapses of non-existing neuron requested");
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



unsigned
NetworkImpl::fractionalBits() const
{
	/* In the worst case we may have all presynaptic neurons for some neuron
	 * firing, and having all the relevant synapses have the maximum weight we
	 * just computed. Based on this, it's possible to set the radix point such
	 * that we are guaranteed never to overflow. However, if we optimise for
	 * this pathological case we'll end up throwing away precision for no
	 * appreciable gain. Instead we rely on overflow detection on the CUDA
	 * device (which will lead to saturation of the input current).
	 *
	 * We can make some reasonable assumptions regarding the number of neurons
	 * expected to fire at any time as well as the distribution of weights.
	 *
	 * For now just assume that at most a fixed number of neurons will fire at
	 * max weight. */
	//! \todo do this based on both max weight and max number of incoming synapses
	float maxAbsWeight = std::max(abs(minWeight()), abs(maxWeight()));
	unsigned log2Ceil = unsigned(ceilf(log2f(maxAbsWeight)));
	unsigned fbits = 31 - log2Ceil - 5; // assumes max 2^5 incoming spikes with max weight
	return fbits;
}


}
