/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/thread.hpp>

#include <nemo/config.h>

#include "Worker.hpp"
#include "Simulation.hpp"
#ifdef CAN_SET_THREAD_AFFINITY
#include "thread_affinity.h"
#endif

namespace nemo {
	namespace cpu {

Worker::Worker(unsigned id, size_t jobSize, size_t neuronCount, Simulation* sim) :
	m_start(id * jobSize),
	m_end(std::min((id+1) * jobSize, neuronCount)),
	m_id(id), // ! \todo just use a static for this
	m_cores(boost::thread::hardware_concurrency()),
	m_sim(sim)
{
	;
}


void
Worker::operator()()
{
#ifdef CAN_SET_THREAD_AFFINITY
	/* The affinity can only be set once the thread has started */
	setThreadAffinity(m_id % m_cores);
#endif
	m_sim->updateRange(m_start, m_end);
}

}	} // end namespaces
