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
#include <boost/mpi/environment.hpp>

#include <ConnectivityMatrix.hpp>
#include "mpi_types.hpp"

namespace nemo {

	class NetworkImpl;
	class ConfigurationImpl;

	namespace mpi {

	class Mapper;
	class SpikeQueue;

/*
 * prefixes:
 * 	l/g distinguishes local/global
 * 	i/o distinguishes input/output (for global)
 */
class Worker
{
	public:

		Worker(boost::mpi::environment& env,
				boost::mpi::communicator& world);

		typedef std::vector<unsigned> fbuf;

	private:

		//! \todo move this to common types
		typedef int rank_t;

		typedef Synapse<unsigned, unsigned, float> synapse_t;

		/* Most of the data for the global synapses will be stored at the node
		 * where the target neuron is processed. We need to build up the
		 * collection of these synapses and later send these to the target
		 * node.
		 *
		 * We could potentially interleave this with the other construction */
		typedef std::map<rank_t, std::vector<SynapseVector> > global_fcm_t;

		/* Buffer for incoming/outgoing data */
		//! \todo make this a non-member
		std::vector<SynapseVector> m_ibuf;
		std::vector<SynapseVector> m_obuf;
		
		void addSynapseVector(const Mapper&, nemo::NetworkImpl& net, global_fcm_t&);
		void addNeuron(nemo::NetworkImpl& net);

		void exchangeGlobalData(const Mapper&, global_fcm_t& g_ss, nemo::ConnectivityMatrix& l_fcm);

		boost::mpi::communicator m_world;

		rank_t m_rank;

		/* All the peers to which this worker should send firing data every
		 * simulation cycle */
		std::set<rank_t> mg_targets;

		/* All the peers from which this worker should receive firing data
		 * every simulation cycle */
		std::set<rank_t> mg_sources;

		/* The specific source firings we should send. */
		std::map<nidx_t, std::set<rank_t> > mg_fcm;

		unsigned ml_scount;
		unsigned mgi_scount;
		unsigned mgo_scount;
		unsigned m_ncount;

		typedef std::vector<boost::mpi::request> req_vector;
		typedef std::map<rank_t, fbuf> fbuf_vector;

		void runSimulation(const nemo::NetworkImpl& net,
				const nemo::ConfigurationImpl& conf,
				const nemo::ConnectivityMatrix& l_fcm,
				size_t localCount);

		void initGlobalScatter(const fbuf& fired, req_vector& oreqs, fbuf_vector& obufs);
		void waitGlobalScatter(req_vector&);

		void initGlobalGather(req_vector& ireqs, fbuf_vector& ibufs);

		void waitGlobalGather(req_vector& ireqs,
				const fbuf_vector& ibufs,
				const nemo::ConnectivityMatrix& l_fcm,
				SpikeQueue& queue);

		void globalGather(const nemo::ConnectivityMatrix& l_fcm, SpikeQueue& queue);

		void sendMaster(const fbuf& fired);
};


}	}

#endif
