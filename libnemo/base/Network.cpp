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


void
Network::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char plastic)
{
	m_impl->addSynapse(source, target, delay, weight, plastic);
}


void
Network::addSynapses(
		unsigned source,
		const std::vector<unsigned>& targets,
		const std::vector<unsigned>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char>& plastic)
{
	m_impl->addSynapses(source, targets, delays, weights, plastic);	
}



template
void
Network::addSynapses<unsigned, unsigned, float, unsigned char>(unsigned,
		const unsigned[], const unsigned[], const float[],
		const unsigned char[], size_t);


template<typename N, typename D, typename W, typename B>
void
Network::addSynapses(
		N source,
		const N targets[],
		const D delays[],
		const W weights[],
		const B plastic[],
		size_t length)
{
	m_impl->addSynapses<N, D, W, B>(source,
			targets, delays, weights, plastic, length);
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
