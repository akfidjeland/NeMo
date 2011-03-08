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
#include <deque>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <nemo/config.h>
#include <nemo/types.hpp>
#include <nemo/Timer.hpp>
#include <nemo/network/Generator.hpp>

#include "Mapper.hpp"
#ifdef NEMO_MPI_DEBUG_TIMING
#	include "MpiTimer.hpp"
#endif

namespace nemo {

	class Network;
	class Configuration;
	namespace network {
		class NetworkImpl;
	}

	namespace mpi {


class Master
{
	public :

		Master( boost::mpi::environment& env,
				boost::mpi::communicator& world,
				const Network&, 
				const Configuration&);

		~Master();

		void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());

		/* Return reference to first buffered cycle's worth of firing. The
		 * reference is invalidated by any further calls to readFiring, or to
		 * step. */
		const std::vector<unsigned>& readFiring();

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

	private :

		boost::mpi::communicator m_world;

		Mapper m_mapper;

		unsigned workers() const;

		void terminate();

		//! \todo use FiringBuffer here instead
		std::deque< std::vector<unsigned> > m_firing;

		void distributeSynapses(const Mapper& mapper, const network::Generator& net);
		void distributeNeurons(const Mapper& mapper, const network::Generator& net);

		Timer m_timer;

#ifdef NEMO_MPI_DEBUG_TIMING
		MpiTimer m_mpiTimer;
#endif
};

	} // end namespace mpi
} // end namespace nemo

#endif
