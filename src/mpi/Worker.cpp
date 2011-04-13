/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Worker.hpp"

#include <algorithm>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include <nemo/internals.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/config.h>

#include "Mapper.hpp"
#ifdef NEMO_MPI_DEBUG_TIMING
#	include "MpiTimer.hpp"
#endif
#include "SpikeQueue.hpp"
#include "nemo_mpi_common.hpp"
#include "types.hpp"
#include "log.hpp"


namespace nemo {
	namespace mpi {


nemo::Configuration
getConfiguration(boost::mpi::communicator& world)
{
	ConfigurationImpl conf;
	boost::mpi::broadcast(world, conf, MASTER);
	conf.disableLogging();
	if(!conf.fractionalBitsSet()) {
		throw nemo::exception(NEMO_UNKNOWN_ERROR, "Fractional bits not set when using MPI backend");
	}
	return Configuration(conf, true); // this will override backend
}



Mapper
getMapper(boost::mpi::communicator& world)
{
	//! \todo fold this into the Mapper ctor instead
	unsigned neurons;
	boost::mpi::broadcast(world, neurons, MASTER);
	int workers = world.size() - 1;
	return Mapper(neurons, workers, world.rank());
}



void
runWorker(boost::mpi::environment& env,
		boost::mpi::communicator& world)
{
	MPI_LOG("Starting worker %u on %s\n", world.rank(), env.processor_name().c_str());
	Configuration conf = getConfiguration(world);
	MPI_LOG("Worker %u using %s\n", world.rank(), conf.backendDescription().c_str());
	MPI_LOG("Worker %u: Creating mapper\n", world.rank());
	Mapper mapper = getMapper(world);
	MPI_LOG("Worker %u: Creating runtime data\n", world.rank());
	try {
		Worker sim(world, conf, mapper);
	} catch (nemo::exception& e) {
		std::cerr << world.rank() << ":" << e.what() << std::endl;
		env.abort(e.errorNumber());
	} catch (std::exception& e) {
		std::cerr << world.rank() << ": " << e.what() << std::endl;
		env.abort(-1);
	}
}



Worker::Worker(
		boost::mpi::communicator& world,
		Configuration& conf,
		Mapper& globalMapper) :
	m_world(world),
	m_rank(world.rank()),
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	m_packetsSent(0),
	m_packetsReceived(0),
	m_bytesSent(0),
	m_bytesReceived(0),
#endif
	ml_scount(0),
	mgi_scount(0),
	mgo_scount(0),
	m_ncount(0)
{
	MPI_LOG("Worker %u: constructing network\n", m_rank);

	/* Temporary network, used to initialise backend */
	network::NetworkImpl net;

	loadNeurons(net);

	/* Global synapses */
	std::deque<Synapse> globalSynapses;
	loadSynapses(globalMapper, globalSynapses, net);

	MPI_LOG("Worker %u: %u neurons\n", m_rank, m_ncount);
	MPI_LOG("Worker %u: %u local synapses\n", m_rank, ml_scount);
	MPI_LOG("Worker %u: %u global synapses (out)\n", m_rank,  mgo_scount);
	MPI_LOG("Worker %u: %u global synapses (int)\n", m_rank, mgi_scount);

	//! \todo move all intialisation into ctor, and make run a separate function.
	runSimulation(globalSynapses, net, *conf.m_impl, globalMapper.neuronCount());
}



void
Worker::loadNeurons(network::NetworkImpl& net)
{
	std::vector<network::Generator::neuron> neurons;
	while(true) {
		int tag;
		broadcast(m_world, tag, MASTER);
		if(tag == NEURON_VECTOR) {
			scatter(m_world, neurons, MASTER);
			for(std::vector<network::Generator::neuron>::const_iterator n = neurons.begin();
					n != neurons.end(); ++n) {
				net.addNeuron(n->first, n->second);
				m_ncount++;
			}
		} else if(tag == NEURONS_END) {
			break;
		} else {
			throw nemo::exception(NEMO_MPI_ERROR, "Unknown tag received during neuron scatter");
		}
	}
}



void
Worker::loadSynapses(
		const Mapper& mapper,
		std::deque<Synapse>& globalSynapses,
		network::NetworkImpl& net)
{
	while(true) {
		int tag;
		broadcast(m_world, tag, MASTER);
		if(tag == SYNAPSE_VECTOR) {
			std::vector<Synapse> ss;
			scatter(m_world, ss, MASTER);
			for(std::vector<Synapse>::const_iterator s = ss.begin(); s != ss.end(); ++s) {
				addSynapse(*s, mapper, globalSynapses, net);
			}
		} else if(tag == SYNAPSES_END) {
			break;
		} else {
			throw nemo::exception(NEMO_MPI_ERROR, "Unknown tag received during synapse scatter");
		}
	}
}



//! \todo could use an iterator which sets up local simulations as a side effect
// we can pass this iterator to the local simulation and incrementally construct our own CM



void
Worker::addSynapse(const Synapse& s,
		const Mapper& mapper,
		std::deque<Synapse>& globalSynapses,
		network::NetworkImpl& net)
{
	const int sourceRank = mapper.rankOf(s.source);
	const int targetRank = mapper.rankOf(s.target());

	if(sourceRank == targetRank) {
		/* Most neurons should be purely local neurons */
		assert(sourceRank == m_rank); // see how master performs seneding
		/*! \todo could use a function to pass in synapse directly instead of
		 * constructing an intermediate network. */
		net.addSynapse(s.source, s.target(), s.delay, s.weight(), s.plastic());
		ml_scount++;
	} else if(sourceRank == m_rank) {
		/* Source neuron is found on this node, but target is on some other node */
		m_fcmOut[s.source].insert(targetRank);
		//! \todo could construct mg_targetNodes after completion of m_fcmOut
		mg_targetNodes.insert(targetRank);
		mgo_scount++;
	} else if(targetRank == m_rank) {

		/* Source neuron is found on some other node, but target neuron is
		 * found here. Incoming spikes are handled by the worker, before being
		 * handed over to the workers underlying simulation */
		mgi_scount++;
		mg_sourceNodes.insert(sourceRank);

		/* Just save the synapse for now. The relevant FCM object is
		 * constructed later, once we know how the local mapping should be done */
		globalSynapses.push_back(s);
	}
}




void
gather(const SpikeQueue& queue,
		const nemo::ConnectivityMatrix& fcm,
		std::vector<fix_t>& current)
{
	std::fill(current.begin(), current.end(), 0.0);

	SpikeQueue::const_iterator arrival_end = queue.current_end();
	for(SpikeQueue::const_iterator arrival = queue.current_begin();
			arrival != arrival_end; ++arrival) {
		const Row& row = fcm.getRow(arrival->source(), arrival->delay());
		FAxonTerminal* row_end = row.data.get() + row.len;
		for(FAxonTerminal* terminal = row.data.get(); terminal != row_end; ++terminal) {
			current.at(terminal->target) += terminal->weight;
		}
	}
}



#ifdef NEMO_MPI_DEBUG_TIMING
#define STEP(name, code)                                                      \
    MPI_LOG("c%u: worker %u %s\n", cycle, m_rank, name);                      \
    code;                                                                     \
    timer.substep()
#else
#define STEP(name, code)                                                      \
    MPI_LOG("c%u: worker %u %s\n", cycle, m_rank, name);                      \
    code
#endif


void
Worker::runSimulation(
		const std::deque<Synapse>& globalSynapses,
		const network::NetworkImpl& net,
		const nemo::ConfigurationImpl& conf,
		unsigned localCount)
{
	MPI_LOG("Worker %u starting simulation\n", m_rank);

	/* Local simulation data */
	boost::scoped_ptr<nemo::SimulationBackend> sim(nemo::simulationBackend(net, conf));

	/*! \bug the mapper() method no longer exists (due to more complex mapping
	 * in the different backends). */
	nemo::Mapper<nidx_t, nidx_t>& localMapper = sim->mapper();
	nemo::ConnectivityMatrix g_fcmIn(conf, localMapper);
	for(std::deque<Synapse>::const_iterator s = globalSynapses.begin();
			s != globalSynapses.end(); ++s) {
		g_fcmIn.addSynapse(s->source, localMapper.localIdx(s->target()), *s);
	}
	g_fcmIn.finalize(localMapper, false);

	std::vector<fix_t> istim(localMapper.neuronsInValidRange(), 0);
	assert(localMapper.neuronsInValidRange() >= localCount);
	SpikeQueue queue(net.maxDelay()); // input from global spikes

	/* Incoming master request */
	boost::mpi::request mreq;
	SimulationStep masterReq;

	/* Return data to master */
	boost::mpi::request moreq;

	/* Incoming peer requests */
	//! \todo can just fill this in as we go
	fbuf_vector ibufs;
	req_list ireqs;
	for(std::set<rank_t>::const_iterator i = mg_sourceNodes.begin();
			i != mg_sourceNodes.end(); ++i) {
		ibufs[*i] = fbuf();
	}

	/* Outgoing peer requests. Not all peers are necessarily potential targets
	 * for local neurons, so the output buffer could in principle be smaller.
	 * Allocated it with potentially unused entries so that insertion (which is
	 * frequent) is faster */
	fbuf_vector obufs;
	req_list oreqs;
	for(std::set<rank_t>::const_iterator i = mg_targetNodes.begin();
			i != mg_targetNodes.end(); ++i) {
		obufs[*i] = fbuf();
	}

	/* Everyone should have set up the local simulation now */
	m_world.barrier();

	/* Scatter empty firing packages to start with */
	initGlobalScatter(oreqs, obufs);

#ifdef NEMO_MPI_DEBUG_TIMING
	/* For basic profiling, time the different stages of the main step loop.
	 * Note that the MPI timers we use here are wallclock-timers, and are thus
	 * sensitive to OS effects */
	MpiTimer timer;
#endif

	while(true) {
#ifdef NEMO_MPI_DEBUG_TRACE
		unsigned cycle = sim->elapsedSimulation();
#endif

		STEP("init incoming master req", mreq = m_world.irecv(MASTER, MASTER_STEP, masterReq));

		/*! \note could use globalGather instead of initGlobalGather/waitGlobalGather */
		// globalGather(l_fcm, queue);
		
		STEP("init global gather", initGlobalGather(ireqs, ibufs));
		//! \todo local gather
		STEP("wait global scatter", waitGlobalScatter(oreqs));
		STEP("global gather", waitGlobalGather(ireqs, ibufs, g_fcmIn, queue));
		STEP("enqueue", enqueAllIncoming(ibufs, g_fcmIn, queue));
		//! \todo improve naming
		//! \todo experiment with order of gather and mreq
		STEP("local gather", gather(queue, g_fcmIn, istim));
		STEP("wait incoming master req", mreq.wait());
		if(masterReq.terminate) {
			break;
		}
		STEP("firing stimulus", sim->setFiringStimulus(masterReq.fstim));
		STEP("current stimulus", sim->setCurrentStimulus(istim));
		//! \todo split up step and only do neuron update here
		STEP("gather (kernel)", sim->prefire());
		STEP("step", sim->fire());
		STEP("scatter (kernel)", sim->postfire());
		STEP("read firing", FiredList fired = sim->readFiring());
		//! \note take care here: fired contains reference to internal buffers in sim.
		STEP("buffer scatter data", bufferScatterData(fired.neurons, obufs));
		STEP("init global scatter", initGlobalScatter(oreqs, obufs));
		STEP("send master", gather(m_world, fired.neurons, MASTER));
		queue.step();
#ifdef NEMO_MPI_DEBUG_TIMING
		timer.step();
#endif
		//! \todo local scatter
	}

#ifdef NEMO_MPI_DEBUG_TIMING
	timer.report(m_rank);
#endif

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	reportCommunicationCounters();
#endif
}


#undef STEP


#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
void
Worker::reportCommunicationCounters() const
{
	printf("Worker %u\n\tsent %lu packets/%luB (%luB/packet)\n\treceived %lu packets / %luB (%luB/packet)\n",
			m_rank,
			m_packetsSent, m_bytesSent,
			m_packetsSent ? m_bytesSent / m_packetsSent : 0,
			m_packetsReceived, m_bytesReceived,
			m_packetsReceived ? m_bytesReceived / m_packetsReceived : 0);
}
#endif

void
Worker::initGlobalGather(req_list& ireqs, fbuf_vector& ibufs)
{
	assert(mg_sourceNodes.size() == ibufs.size());
	unsigned sid = 0;
	for(std::set<rank_t>::const_iterator source = mg_sourceNodes.begin();
			source != mg_sourceNodes.end(); ++source, ++sid) {
		MPI_LOG("Worker %u init gather from %u\n", m_rank, *source);
		assert(ibufs.find(*source) != ibufs.end());
		fbuf& incoming = ibufs[*source];
		incoming.clear();
		//! \todo do we do an incorrect copy operation here?
		ireqs.push_back(m_world.irecv(*source, WORKER_STEP, incoming));
	}
}



/* Incoming spike/delay pairs to spike queue */
void
enqueueIncoming(
		const Worker::fbuf& fired,
		const nemo::ConnectivityMatrix& cm,
		SpikeQueue& queue)
{
	for(Worker::fbuf::const_iterator i_source = fired.begin();
			i_source != fired.end(); ++i_source) {
		nidx_t source = *i_source;
		typedef nemo::ConnectivityMatrix::delay_iterator it;
		it end = cm.delay_end(source);
		for(it delay = cm.delay_begin(source); delay != end; ++delay) {
			// -1 since spike has already been in flight for a cycle
			queue.enqueue(source, *delay, 1);
		}
	}
}



void
Worker::enqueAllIncoming(
		const fbuf_vector& ibufs,
		const nemo::ConnectivityMatrix& l_fcm,
		SpikeQueue& queue)
{
	for(fbuf_vector::const_iterator i = ibufs.begin(); i != ibufs.end(); ++i) {
#ifdef NEMO_MPI_DEBUG_TRACE
		rank_t source = i->first;
#endif
		const fbuf& fired = i->second;
		MPI_LOG("Worker %u receiving %lu firings from %u\n", m_rank, fired.size(), source);
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
		m_bytesReceived += sizeof(unsigned) * fired.size();
#endif
		enqueueIncoming(fired, l_fcm, queue);
	}
}




/* Wait for all incoming firings sent during the previous cycle. Add these
 * firings to the queue */
void
Worker::waitGlobalGather(
		req_list& ireqs,
		const fbuf_vector& ibufs,
		const nemo::ConnectivityMatrix& l_fcm,
		SpikeQueue& queue)
{
	using namespace boost::mpi;

	MPI_LOG("Worker %u waiting for messages from %lu peers\n", m_rank, ireqs.size());

	unsigned nreqs = ireqs.size();
	for(unsigned r=0; r < nreqs; ++r) {
		std::pair<status, req_list::iterator> result = wait_any(ireqs.begin(), ireqs.end());
		ireqs.erase(result.second);
	}
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	m_packetsReceived += ireqs.size();
#endif
}




/*! \note this function is not currently in use, but can be used in place of
 * the initGlobalGather/waitGlobalGather pair. However, if using a single
 * in-order globalGather we have less opportunities for overlapping
 * communication and computation.  */
void
Worker::globalGather(
		const nemo::ConnectivityMatrix& l_fcm,
		SpikeQueue& queue)
{
	for(std::set<rank_t>::const_iterator source = mg_sourceNodes.begin();
			source != mg_sourceNodes.end(); ++source) {
		fbuf incoming;
		m_world.irecv(*source, WORKER_STEP, incoming);
		MPI_LOG("Worker %u receiving %lu firings from %u\n", m_rank, incoming.size(), *source);
		enqueueIncoming(incoming, l_fcm, queue);
	}

}



/* Sort outgoing firing data into per-node buffers
 *
 * \param fired
 * 		Firing generated this cycle in the local simulation
 * \param obuf
 * 		Per-rank buffer of firing.
 */
void
Worker::bufferScatterData(const fbuf& fired, fbuf_vector& obufs)
{
	for(fbuf_vector::iterator i = obufs.begin(); i != obufs.end(); ++i) {
		i->second.clear();
	}

	/* Each local firing may be sent to zero or more peers */
	for(std::vector<unsigned>::const_iterator source = fired.begin();
			source != fired.end(); ++source) {
		const std::set<rank_t>& targets = m_fcmOut[*source];
		for(std::set<rank_t>::const_iterator target = targets.begin();
				target != targets.end(); ++target) {
			rank_t targetRank = *target;
			assert(mg_targetNodes.count(targetRank) == 1);
			obufs[targetRank].push_back(*source);
		}
	}
}



/* Initialise asynchronous send of firing to all neighbours.
 *
 * \param oreqs
 * 		List of requests which will be populated by this function. Any existing
 * 		contents will be cleared.
 * \param obuf
 * 		Per-rank buffer of firing.
 */
void
Worker::initGlobalScatter(req_list& oreqs, fbuf_vector& obufs)
{
	MPI_LOG("Worker %u sending firing to %lu peers\n", m_rank, mg_targetNodes.size());

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	m_packetsSent += mg_targetNodes.size();
#endif

	oreqs.clear();

	for(std::set<rank_t>::const_iterator target = mg_targetNodes.begin();
			target != mg_targetNodes.end(); ++target) {
		rank_t targetRank = *target;
		MPI_LOG("Worker %u sending %lu firings to %u\n", m_rank, obufs[targetRank].size(), targetRank);
		oreqs.push_back(m_world.isend(targetRank, WORKER_STEP, obufs[targetRank]));
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
		m_bytesSent += sizeof(unsigned) * obufs[targetRank].size();
#endif
	}
}



void
Worker::waitGlobalScatter(req_list& oreqs)
{
	boost::mpi::wait_all(oreqs.begin(), oreqs.end());
}



	} // end namespace mpi
} // end namespace nemo
