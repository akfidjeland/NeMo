#include "Network.hpp"
#include "NetworkImpl.hpp"

namespace nemo {

Network::Network() :
	m_impl(new network::NetworkImpl())
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
