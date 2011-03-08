#include "Configuration.hpp"
#include "ConfigurationImpl.hpp"

#include <boost/format.hpp>
#include <nemo/internals.hpp>
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

Configuration::Configuration() :
	m_impl(new ConfigurationImpl())
{
	setDefaultHardware(*m_impl);
	setBackendDescription();
}



Configuration::Configuration(const Configuration& other) :
	m_impl(new ConfigurationImpl(*other.m_impl))
{
	;
}


Configuration::Configuration(const ConfigurationImpl& other, bool ignoreBackendOptions) :
	m_impl(new ConfigurationImpl(other))
{
	if(ignoreBackendOptions) {
		setDefaultHardware(*m_impl);
		setBackendDescription();
	}
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
Configuration::setStdpFunction(
				const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minWeight,
				float maxWeight)
{
	m_impl->setStdpFunction(prefire, postfire, minWeight, maxWeight);
}



void
Configuration::setWriteOnlySynapses()
{
	m_impl->setWriteOnlySynapses();
}


bool
Configuration::writeOnlySynapses() const
{
	return m_impl->writeOnlySynapses();
}



void
Configuration::setCpuBackend(int threadCount)
{
	// try setting the CUDA backend
	cpu::chooseHardwareConfiguration(*m_impl, threadCount);
	setBackendDescription();
}


void
Configuration::setCudaBackend(int device)
{
	setCudaDeviceConfiguration(*m_impl, device);
	setBackendDescription();
}



backend_t
Configuration::backend() const
{
	return m_impl->backend();
}



int
Configuration::cudaDevice() const
{
	if(m_impl->backend() == NEMO_BACKEND_CUDA) {
		return m_impl->cudaDevice();
	} else {
		return -1;
	}
}


int
Configuration::cpuThreadCount() const
{
	if(m_impl->backend() == NEMO_BACKEND_CPU) {
		return m_impl->cpuThreadCount();
	} else {
		return -1;
	}
}



const char*
Configuration::backendDescription() const
{
	return m_impl->backendDescription();
}



void
Configuration::setBackendDescription()
{
	using boost::format;

	unsigned tcount;
	switch(m_impl->backend()) {
		case NEMO_BACKEND_CUDA :
			m_impl->setBackendDescription(cudaDeviceDescription(m_impl->cudaDevice()));
			break;
		case NEMO_BACKEND_CPU :
			//! \todo throw from cpuThreadCount. Set post condition
			tcount = m_impl->cpuThreadCount();
			if(tcount == 1) {
				m_impl->setBackendDescription("CPU backend (single-threaded)");
			} else {
				m_impl->setBackendDescription(str(format("CPU backend (%u threads)") % tcount).c_str());
			}
			break;
		default :
			throw std::runtime_error("Invalid backend selected");
	}
}


} // end namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::Configuration const& conf)
{
	return o << *conf.m_impl;
}
