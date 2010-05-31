#include "Configuration.hpp"
#include "ConfigurationImpl.hpp"

namespace nemo {

Configuration::Configuration() :
	m_impl(new ConfigurationImpl())
{
	;
}


Configuration::~Configuration()
{
	delete m_impl;
}


void
Configuration::enableLogging()
{
	m_impl->enableLogging();
}


void
Configuration::disableLogging()
{
	m_impl->disableLogging();
}


bool
Configuration::loggingEnabled() const
{
	return m_impl->loggingEnabled();
}


void
Configuration::setCudaPartitionSize(unsigned ps)
{
	m_impl->setCudaPartitionSize(ps);
}


unsigned
Configuration::cudaPartitionSize() const
{
	return m_impl->cudaPartitionSize();
}


void
Configuration::setCudaFiringBufferLength(unsigned cycles)
{
	m_impl->setCudaFiringBufferLength(cycles);
}


unsigned
Configuration::cudaFiringBufferLength() const
{
	return m_impl->cudaFiringBufferLength();
}


int
Configuration::setCudaDevice(int dev)
{
	return m_impl->setCudaDevice(dev);
}


void
Configuration::setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight)
{
	m_impl->setStdpFunction(prefire, postfire, minWeight, maxWeight);
}

} // end namespace nemo



std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf)
{
	return o << *conf.m_impl;
}
