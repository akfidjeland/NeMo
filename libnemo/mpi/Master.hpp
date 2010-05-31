#ifndef NEMO_MPI_MASTER_HPP
#define NEMO_MPI_MASTER_HPP

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

	class Network;
	class NetworkImpl;
	class Configuration;

	namespace mpi {


class Master
{
	public :

		Master(boost::mpi::communicator& world,
				const Network&, 
				const Configuration&);

		~Master();

		void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());

	private :

		boost::mpi::communicator m_world;

		void distributeNetwork(nemo::NetworkImpl* net);

		unsigned workers() const;

		void terminate();
};

	} // end namespace mpi
} // end namespace nemo

#endif
