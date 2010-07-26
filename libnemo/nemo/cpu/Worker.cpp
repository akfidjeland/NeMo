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

#include <boost/thread.hpp>

#include "Worker.hpp"
#include "Simulation.hpp"

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
	//! \todo add cmake configuration for setting thread affinity
	/* The affinity can only be set once the thread has started */
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(m_id % m_cores, &cpuset);
	pid_t tid = syscall(__NR_gettid);
	sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
	m_sim->updateRange(m_start, m_end);
}

}	} // end namespaces
