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



const std::vector<unsigned>&
Network::getTargets(unsigned source) const
{
	return m_impl->getTargets(source);
}



const std::vector<unsigned>&
Network::getDelays(unsigned source) const
{
	return m_impl->getDelays(source);
}



const std::vector<float>&
Network::getWeights(unsigned source) const
{
	return m_impl->getWeights(source);
}



const std::vector<unsigned char>&
Network::getPlastic(unsigned source) const
{
	return m_impl->getPlastic(source);
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
