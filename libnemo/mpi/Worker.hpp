#ifndef NEMO_MPI_WORKER_HPP
#define NEMO_MPI_WORKER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/mpi/communicator.hpp>
#include <types.hpp>

namespace nemo {

	class NetworkImpl;

	namespace mpi {

	class Mapper;

class Worker
{
	public:

		Worker(boost::mpi::communicator& world);

	private:

		//! \todo add a local simulation here

		/* Buffer for incoming data */
		std::vector<Synapse<unsigned, unsigned, float> > m_ss;
		
		void addSynapseVector(nemo::NetworkImpl& net, const Mapper&);

		boost::mpi::communicator m_world;

		int m_rank;
};

	}
}

#endif
