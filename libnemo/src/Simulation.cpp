/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <Simulation.hpp>
#include "CudaSimulation.hpp"
#include <Network.hpp>
#ifdef INCLUDE_TIMING_API
#include "Timer.hpp"
#endif

namespace nemo {


Simulation*
Simulation::create(const Network& net, const Configuration& conf)
{
	if(net.neuronCount() == 0) {
		throw std::runtime_error("Cannot create simulation from empty network");
		return NULL;
	}
	int dev = cuda::Simulation::selectDevice();
	return dev == -1 ? NULL : new cuda::Simulation(*(net.m_impl), conf);
}


Simulation::Simulation()
#ifdef INCLUDE_TIMING_API
	: m_timer(new Timer())
#endif
{
	;
}


Simulation::~Simulation()
{
#ifdef INCLUDE_TIMING_API
	delete m_timer;
#endif
}



void
Simulation::stepTimer()
{
#ifdef INCLUDE_TIMING_API
	m_timer->step();
#endif
}



unsigned long
Simulation::elapsedWallclock() const
{
#ifdef INCLUDE_TIMING_API
	return m_timer->elapsedWallclock();
#else
	throw std::runtime_error("elapsedWallclock is not supported in this version");
#endif
}



unsigned long
Simulation::elapsedSimulation() const
{
#ifdef INCLUDE_TIMING_API
	return m_timer->elapsedSimulation();
#else
	throw std::runtime_error("elapsedSimulation is not supported in this version");
#endif
}



void
Simulation::resetTimer()
{
#ifdef INCLUDE_TIMING_API
	m_timer->reset();
#endif
}


} // end namespace nemo
