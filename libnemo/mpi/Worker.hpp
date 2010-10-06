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
#include <deque>
#include <set>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>

namespace nemo {

	namespace network {
		class NetworkImpl;
	}
	//! \todo use consistent interface here
	class ConfigurationImpl;
	class Configuration;
	class ConnectivityMatrix;

	namespace mpi {

	class Mapper;
	class SpikeQueue;


void
runWorker(boost::mpi::environment& env, boost::mpi::communicator& world);


/*
 * prefixes:
 * 	l/g distinguishes local/global
 * 	i/o distinguishes input/output (for global)
 */
class Worker
{
	public:

		Worker( boost::mpi::communicator& world,
				Configuration& conf,
				Mapper& mapper);

		//! \todo move this type to nemo::FiringBuffer instead perhaps typedefed as Fired::neuron_list
		typedef std::vector<unsigned> fbuf;

	private:

		//! \todo move this to common types
		typedef int rank_t;

		/* While most synapses are likely to be local to a single simulation
		 * (i.e. both the source and target neurons are found on the same
		 * node), there will also be a large number of /global/ synapses, which
		 * cross node boundaries. Most of the associated synapse data is stored
		 * at the node containing the target neuron. */

		/* On the node with the source, we only need to store a mapping from
		 * the neuron (in local indices) to the target nodes (rank id). */
		std::map<nidx_t, std::set<rank_t> > m_fcmOut;

		/* On the node with the target we store a connectivity matrix where the
		 * source neurons are specified in global ids, while the targets are
		 * stored in local ids (to simplify forwarding to the local simulation
		 * object at run-time. See runSimulation for the construction of this
		 * object. */

		void loadNeurons(network::NetworkImpl& net);

		void loadSynapses(const Mapper&,
				std::deque<Synapse>& globalSynapses,
				network::NetworkImpl& net);

		void addSynapse(const Synapse& s,
				const Mapper& mapper,
				std::deque<Synapse>& globalSynapses,
				network::NetworkImpl& net);

		boost::mpi::communicator m_world;

		rank_t m_rank;

		/* All the peers to which this worker should send firing data every
		 * simulation cycle */
		//! \todo since we continually update this, could make it a vector<bool> instead
		std::set<rank_t> mg_targetNodes;

		/* All the peers from which this worker should receive firing data
		 * every simulation cycle */
		std::set<rank_t> mg_sourceNodes;

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
		uint64_t m_packetsSent;
		uint64_t m_packetsReceived;
		uint64_t m_bytesSent;
		uint64_t m_bytesReceived;

		void reportCommunicationCounters() const;
#endif

		unsigned ml_scount;
		unsigned mgi_scount;
		unsigned mgo_scount;
		unsigned m_ncount;

		typedef std::list<boost::mpi::request> req_list;
		typedef std::map<rank_t, fbuf> fbuf_vector;

		void runSimulation(
				const std::deque<Synapse>& globalSynapses,
				const network::NetworkImpl& net,
				const nemo::ConfigurationImpl& conf,
				unsigned localCount);

		void bufferScatterData(const fbuf& fired, fbuf_vector& obufs);
		void initGlobalScatter(req_list& oreqs, fbuf_vector& obufs);
		void waitGlobalScatter(req_list&);

		void initGlobalGather(req_list& ireqs, fbuf_vector& ibufs);

		void waitGlobalGather(req_list& ireqs,
				const fbuf_vector& ibufs,
				const nemo::ConnectivityMatrix& l_fcm,
				SpikeQueue& queue);

		void enqueAllIncoming(
				const fbuf_vector& bufs,
				const nemo::ConnectivityMatrix& l_fcm,
				SpikeQueue& queue);

		void globalGather(const nemo::ConnectivityMatrix& l_fcm, SpikeQueue& queue);

};


}	}

#endif
