/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Simulation.hpp"
#include "CudaSimulation.hpp"
#include "Timer.hpp"

namespace nemo {


Simulation*
Simulation::create(const Network& net, const Configuration& conf)
{
	int dev = cuda::Simulation::selectDevice();
	return dev == -1 ? NULL : new cuda::Simulation(net, conf);
}


Simulation::Simulation() :
	m_timer(new Timer())
{
	;
}


Simulation::~Simulation()
{
	delete m_timer;
}


#ifdef INCLUDE_TIMING_API


void
Simulation::stepTimer()
{
	m_timer->step();
}


unsigned long
Simulation::elapsedWallclock() const
{
	return m_timer->elapsedWallclock();
}



unsigned long
Simulation::elapsedSimulation() const
{
	return m_timer->elapsedSimulation();
}



void
Simulation::resetTimer()
{
	m_timer->reset();
}

#endif

} // end namespace nemo
