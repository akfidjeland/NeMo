/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "CudaSimulation.hpp"
#include "CudaSimulationImpl.hpp"


namespace nemo {
	namespace cuda {

Simulation::Simulation(
		const nemo::NetworkImpl& net,
		const nemo::ConfigurationImpl& conf) :
	m_impl(new SimulationImpl(net, conf))
{
	resetTimer();
}


Simulation::~Simulation()
{
	delete m_impl;
}


int
Simulation::selectDevice()
{
	return SimulationImpl::selectDevice();
}


int
Simulation::setDevice(int dev)
{
	return SimulationImpl::setDevice(dev);
}


unsigned
Simulation::getFiringBufferLength() const
{
	return m_impl->getFiringBufferLength();
}


void
Simulation::step(const std::vector<unsigned>& fstim)
{
	m_impl->step(fstim);
}


void
Simulation::applyStdp(float reward)
{
	m_impl->applyStdp(reward);		
}


void
Simulation::getSynapses(unsigned source,
		const std::vector<unsigned>** targets,
		const std::vector<unsigned>** delays,
		const std::vector<float>** weights,
		const std::vector<unsigned char>** plastic)
{
	m_impl->getSynapses(source, targets, delays, weights, plastic);
}


unsigned
Simulation::readFiring(const std::vector<unsigned>** cycles,
		const std::vector<unsigned>** nidx)
{
	return m_impl->readFiring(cycles, nidx);
}


void
Simulation::flushFiringBuffer()
{
	m_impl->flushFiringBuffer();
}


void
Simulation::finishSimulation()
{
	m_impl->finishSimulation();
}


unsigned long
Simulation::elapsedWallclock() const
{
	return m_impl->elapsedWallclock();
}


unsigned long
Simulation::elapsedSimulation() const
{
	return m_impl->elapsedSimulation();
}


void
Simulation::resetTimer()
{
	m_impl->resetTimer();
}


unsigned
Simulation::defaultPartitionSize()
{
	return SimulationImpl::defaultPartitionSize();
}


unsigned
Simulation::defaultFiringBufferLength()
{
	return SimulationImpl::defaultFiringBufferLength();
}

	} // end namespace cuda
} // end namespace nemo
