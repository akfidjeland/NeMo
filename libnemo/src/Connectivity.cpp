/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Connectivity.hpp"

#include <assert.h>
#include <stdexcept>

namespace nemo {


Connectivity::Connectivity() :
	m_maxSourceIdx(0),
	m_maxDelay(0),
	m_maxWeight(0),
	m_minWeight(0)
{ }


void
Connectivity::addNeuron(nidx_t nidx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	if(m_neurons.find(nidx) != m_neurons.end()) {
		//! \todo construct a sensible error message here using sstream
		throw std::runtime_error("duplicate neuron index");
	}
	m_neurons[nidx] = Neuron<float>(a, b, c, d, u, v, sigma);
}



void
Connectivity::addSynapse(
		nidx_t source,
		nidx_t target,
		delay_t delay,
		weight_t weight,
		uchar plastic)
{
	m_fcm[source][delay].push_back(synapse_t(target, weight, plastic));

	//! \todo make sure we don't have maxDelay in cuda::Connectivity
	m_maxSourceIdx = std::max(m_maxSourceIdx, source);
	m_maxDelay = std::max(m_maxDelay, delay);
	m_maxWeight = std::max(m_maxWeight, weight);
	m_minWeight = std::min(m_minWeight, weight);
}


void
Connectivity::addSynapses(
		nidx_t source,
		const std::vector<nidx_t>& targets,
		const std::vector<delay_t>& delays,
		const std::vector<weight_t>& weights,
		const std::vector<uchar> plastic)
{
	size_t length = targets.size();
	//! \todo use exceptions here
	assert(length == delays.size());
	assert(length == weights.size());
	assert(length == plastic.size());

    if(length == 0) {
        return;
	}

	for(size_t i=0; i < length; ++i) {
		addSynapse(source, targets[i], delays[i], weights[i], plastic[i]);
	}
}


}
