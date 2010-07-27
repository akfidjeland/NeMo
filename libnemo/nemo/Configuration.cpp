#include "Configuration.hpp"
#include "ConfigurationImpl.hpp"

#ifdef NEMO_CUDA_ENABLED
#include <nemo/cuda/CudaSimulation.hpp>
#endif
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

Configuration::Configuration() :
	m_impl(new ConfigurationImpl())
{
#ifdef NEMO_CUDA_ENABLED
	m_impl->setCudaPartitionSize(cuda::Simulation::defaultPartitionSize());
	m_impl->setCudaFiringBufferLength(cuda::Simulation::defaultFiringBufferLength());
#endif
	m_impl->setCpuThreadCount(cpu::Simulation::defaultThreadCount());
}



Configuration::Configuration(const Configuration& other) :
	m_impl(new ConfigurationImpl(*other.m_impl))
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
Configuration::setCpuThreadCount(unsigned threads)
{
	m_impl->setCpuThreadCount(threads);
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
#ifdef NEMO_CUDA_ENABLED
	return cuda::Simulation::setDevice(dev);
#else
	return -1;
#endif
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


void
Configuration::setFractionalBits(unsigned bits)
{
	m_impl->setFractionalBits(bits);
}


void
Configuration::setBackend(backend_t backend)
{
	m_impl->setBackend(backend);
}


} // end namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf)
{
	return o << *conf.m_impl;
}
