#include "Network.hpp"
#include "NetworkImpl.hpp"
#include "synapse_indices.hpp"

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



void
Network::setNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	m_impl->setNeuron(idx, a, b, c, d, u, v, sigma);
}



float
Network::getNeuronState(unsigned neuron, unsigned var) const
{
	return m_impl->getNeuronState(neuron, var);
}



float
Network::getNeuronParameter(unsigned neuron, unsigned parameter) const
{
	return m_impl->getNeuronParameter(neuron, parameter);
}



void
Network::setNeuronState(unsigned neuron, unsigned var, float val)
{
	return m_impl->setNeuronState(neuron, var, val);
}



void
Network::setNeuronParameter(unsigned neuron, unsigned parameter, float val)
{
	return m_impl->setNeuronParameter(neuron, parameter, val);
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



unsigned
Network::getSynapseTarget(const synapse_id& id) const
{
	return m_impl->getSynapseTarget(id);
}



unsigned
Network::getSynapseDelay(const synapse_id& id) const
{
	return m_impl->getSynapseDelay(id);
}



float
Network::getSynapseWeight(const synapse_id& id) const
{
	return m_impl->getSynapseWeight(id);
}



unsigned char
Network::getSynapsePlastic(const synapse_id& id) const
{
	return m_impl->getSynapsePlastic(id);
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
