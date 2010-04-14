/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Network.hpp"

#include <stdexcept>
#include <sstream>

namespace nemo {


Network::Network() :
	m_maxSourceIdx(0),
	m_maxDelay(0),
	m_maxWeight(0),
	m_minWeight(0)
{ }


void
Network::addNeuron(unsigned nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	if(m_neurons.find(nidx) != m_neurons.end()) {
		std::ostringstream msg;
		msg << "Duplicate neuron index for neuron " << nidx;
		throw std::runtime_error(msg.str());
	}
	m_neurons[nidx] = Neuron<float>(a, b, c, d, u, v, sigma);
}



void
Network::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		uchar plastic)
{
	m_fcm[source][delay].push_back(synapse_t(target, weight, plastic));

	//! \todo make sure we don't have maxDelay in cuda::ConnectivityMatrix
	m_maxSourceIdx = std::max(m_maxSourceIdx, source);
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);
}



void
Network::addSynapses(
		unsigned source,
		const std::vector<unsigned>& targets,
		const std::vector<unsigned>& delays,
		const std::vector<float>& weights,
		const std::vector<uchar> plastic)
{
	size_t length = targets.size();

	if(length != delays.size() || length != weights.size() || length != plastic.size()) {
		std::ostringstream msg;
		msg << "Synapse vector length mismatch for neuron " << source;
		throw std::runtime_error(msg.str());
	}

    if(length == 0) {
        return;
	}

	for(size_t i=0; i < length; ++i) {
		addSynapse(source, targets[i], delays[i], weights[i], plastic[i]);
	}
}


}
