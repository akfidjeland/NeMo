#ifndef NEMO_CPU_WORKER_HPP
#define NEMO_CPU_WORKER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

namespace nemo {
	namespace  cpu {

class Simulation;

class Worker
{
	public :

		Worker(unsigned id, size_t jobSize, size_t neuronCount, Simulation* sim);

		void operator()();

	private:

		size_t m_start;
		size_t m_end;
		unsigned m_id;
		unsigned m_cores;
		Simulation* m_sim;
};


}	}	// end namespaces

#endif
