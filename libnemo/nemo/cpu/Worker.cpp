/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \todo add configuration for this
#include <sched.h> // for setting thread affinity
#include <sys/types.h>
#include <sys/syscall.h>


#include "Worker.hpp"
#include "Simulation.hpp"

namespace nemo {
	namespace cpu {

Worker::Worker(int id, size_t start, size_t end, Simulation* sim) :
	m_start(start),
	m_end(end),
	m_id(id), // ! \todo just use a static for this
	m_sim(sim)
{
	;
}


void
Worker::operator()()
{
	//! \todo add cmake configuration for setting thread affinity
	/* The affinity can only be set once the thread has started */
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(m_id, &cpuset);
	pid_t tid = syscall(__NR_gettid);
	sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
	m_sim->updateRange(m_start, m_end);
}

}	} // end namespaces
