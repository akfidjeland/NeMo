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
#include <set>
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

		typedef int rank_t;

		typedef Synapse<unsigned, unsigned, float> synapse_t;

		/* Most of the data for the global synapses will be stored at the node
		 * where the target neuron is processed. We need to build up the collection
		 * of these synapses and later send these to the target node.
		 *
		 * We could potentially interleave this with the other construction */
		typedef std::map<rank_t, std::vector<synapse_t> > global_fcm_t;

		//! \todo add a local simulation here

		/* Buffer for incoming data */
		std::vector<synapse_t> m_ss;
		
		void addSynapseVector(const Mapper&, nemo::NetworkImpl& net, global_fcm_t&);
		void addNeuron(nemo::NetworkImpl& net);

		void exchangeGlobalData(global_fcm_t& g_ss);

		boost::mpi::communicator m_world;

		rank_t m_rank;

		/* All the ranks with which this worker should synchronise every simulation cycle */
		std::set<rank_t> m_targets;

		/* The specific source firings we should send */
		std::map<nidx_t, std::set<rank_t> > m_fcm;

		unsigned ml_scount;
		unsigned mg_scount;
		unsigned m_ncount;
};

	}
}

#endif
