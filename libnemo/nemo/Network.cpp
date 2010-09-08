#include "Network.hpp"
#include "NetworkImpl.hpp"

namespace nemo {

Network::Network() :
	m_impl(new NetworkImpl())
{
	;
}


Network::~Network()
{
	delete m_impl;
}


void
Network::addNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	//! \todo could use internal network type here
	m_impl->addNeuron(idx, a, b, c, d, u, v, sigma);
}


synapse_id
Network::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char plastic)
{
	return m_impl->addSynapse(source, target, delay, weight, plastic);
}


void
Network::addSynapses(
		const std::vector<unsigned>& sources,
		const std::vector<unsigned>& targets,
		const std::vector<unsigned>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char>& plastic)
{
	m_impl->addSynapses(sources, targets, delays, weights, plastic);
}



void
Network::addSynapses(
		const unsigned sources[],
		const unsigned targets[],
		const unsigned delays[],
		const float weights[],
		const unsigned char plastic[],
		size_t length)
{
	m_impl->addSynapses<unsigned, unsigned, float, unsigned char>(sources,
			targets, delays, weights, plastic, length);
}



void
Network::getSynapses(
		unsigned source,
		std::vector<unsigned>& targets,
		std::vector<unsigned>& delays,
		std::vector<float>& weights,
		std::vector<unsigned char>& plastic) const
{
	m_impl->getSynapses(source, targets, delays, weights, plastic);
}



unsigned
Network::maxDelay() const 
{
	return m_impl->maxDelay(); 
}



float
Network::maxWeight() const
{ 
	return m_impl->maxWeight();
}



float
Network::minWeight() const
{ 
	return m_impl->minWeight(); 
}



unsigned 
Network::neuronCount() const
{
	return m_impl->neuronCount();
}


} // end namespace nemo
