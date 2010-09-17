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
#include <nemo/config.h>

#include "Mapper.hpp"
#ifdef NEMO_MPI_DEBUG_TIMING
#	include "MpiTimer.hpp"
#endif
#include "SpikeQueue.hpp"
#include "nemo_mpi_common.hpp"
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
		Mapper& mapper) :
	m_fcmIn(*conf.m_impl, mapper),
	m_world(world),
	m_rank(world.rank()),
	ml_scount(0),
	mgi_scount(0),
	mgo_scount(0),
	m_ncount(0)
{
	MPI_LOG("Worker %u: constructing network\n", m_rank);

	/* Temporary network, used to initialise backend */
	network::NetworkImpl net;

	loadNeurons(mapper, net);
	loadSynapses(mapper, net);

	m_fcmIn.finalize(mapper, false);

	MPI_LOG("Worker %u: %u neurons\n", m_rank, m_ncount);
	MPI_LOG("Worker %u: %u local synapses\n", m_rank, ml_scount);
	MPI_LOG("Worker %u: %u global synapses (out)\n", m_rank,  mgo_scount);
	MPI_LOG("Worker %u: %u global synapses (int)\n", m_rank, mgi_scount);

	//! \todo move all intialisation into ctor, and make run a separate function.
	runSimulation(net, *conf.m_impl, mapper.localCount());
}



void
Worker::loadNeurons(Mapper& mapper, network::NetworkImpl& net)
{
	std::vector<network::Generator::neuron> neurons;
	while(true) {
		int tag;
		broadcast(m_world, tag, MASTER);
		if(tag == NEURON_VECTOR) {
			scatter(m_world, neurons, MASTER);
			for(std::vector<network::Generator::neuron>::const_iterator n = neurons.begin();
					n != neurons.end(); ++n) {
				addNeuron(*n, mapper, net);
			}
		} else if(tag == NEURONS_END) {
			break;
		} else {
			throw nemo::exception(NEMO_MPI_ERROR, "Unknown tag received during neuron scatter");
		}
	}
}



void
Worker::loadSynapses(Mapper& mapper, network::NetworkImpl& net)
{
	while(true) {
		int tag;
		broadcast(m_world, tag, MASTER);
		if(tag == SYNAPSE_VECTOR) {
			std::vector<Synapse> ss;
			scatter(m_world, ss, MASTER);
			for(std::vector<Synapse>::const_iterator s = ss.begin(); s != ss.end(); ++s) {
				addSynapse(*s, mapper, net);
			}
		} else if(tag == SYNAPSES_END) {
			break;
		} else {
			throw nemo::exception(NEMO_MPI_ERROR, "Unknown tag received during synapse scatter");
		}
	}
}



void
Worker::addNeuron(const network::Generator::neuron& n,
		Mapper& mapper,
		network::NetworkImpl& net)
{
	net.addNeuron(n.first, n.second);
	m_ncount++;
	mapper.addGlobal(n.first);
}



//! \todo could use an iterator which sets up local simulations as a side effect
// we can pass this iterator to the local simulation and incrementally construct our own CM



void
Worker::addSynapse(const Synapse& s, const Mapper& mapper, network::NetworkImpl& net)
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
		m_fcmIn.addSynapse(s.source, mapper.localIdx(s.target()), s);
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
Worker::runSimulation(const network::NetworkImpl& net,
		const nemo::ConfigurationImpl& conf,
		unsigned localCount)
{
	MPI_LOG("Worker %u starting simulation\n", m_rank);

	/* Local simulation data */
	boost::scoped_ptr<nemo::SimulationBackend> sim(nemo::simulationBackend(net, conf));

	MPI_LOG("Worker %u starting simulation\n", m_rank);

	std::vector<fix_t> istim(localCount, 0);         // input from global spikes
	SpikeQueue queue(net.maxDelay());

	/* Incoming master request */
	boost::mpi::request mreq;
	SimulationStep masterReq;

	/* Return data to master */
	boost::mpi::request moreq;

	/* Incoming peer requests */
	//! \todo can just fill this in as we go
	fbuf_vector ibufs;
	req_vector ireqs(mg_sourceNodes.size());
	for(std::set<rank_t>::const_iterator i = mg_sourceNodes.begin();
			i != mg_sourceNodes.end(); ++i) {
		ibufs[*i] = fbuf();
	}

	/* Outgoing peer requests. Not all peers are necessarily potential targets
	 * for local neurons, so the output buffer could in principle be smaller.
	 * Allocated it with potentially unused entries so that insertion (which is
	 * frequent) is faster */
	fbuf_vector obufs;
	req_vector oreqs(mg_targetNodes.size());
	for(std::set<rank_t>::const_iterator i = mg_targetNodes.begin();
			i != mg_targetNodes.end(); ++i) {
		obufs[*i] = fbuf();
	}

	/* Everyone should have set up the local simulation now */
	m_world.barrier();

	/* Scatter empty firing packages to start with */
	initGlobalScatter(fbuf(), oreqs, obufs);

#ifdef NEMO_MPI_DEBUG_TIMING
	/* For basic profiling, time the different stages of the main step loop.
	 * Note that the MPI timers we use here are wallclock-timers, and are thus
	 * sensitive to OS effects */
	MpiTimer timer;
#endif

	while(!masterReq.terminate) {
#ifdef NEMO_MPI_DEBUG_TIMING
		unsigned cycle = sim->elapsedSimulation();
#endif
		STEP("init incoming master req", mreq = m_world.irecv(MASTER, MASTER_STEP, masterReq));

		/*! \note could use globalGather instead of initGlobalGather/waitGlobalGather */
		// globalGather(l_fcm, queue);
		
		STEP("init global gather", initGlobalGather(ireqs, ibufs));
		//! \todo local gather
		STEP("wait global scatter", waitGlobalScatter(oreqs));
		STEP("wait global gather", waitGlobalGather(ireqs, ibufs, m_fcmIn, queue));
		//! \todo improve naming
		//! \todo experiment with order of gather and mreq
		STEP("gather", gather(queue, m_fcmIn, istim));
		STEP("wait incoming master req", mreq.wait());
		STEP("firing stimulus", sim->setFiringStimulus(masterReq.fstim));
		STEP("current stimulus", sim->setCurrentStimulus(istim));
		//! \todo split up step and only do neuron update here
		STEP("step", sim->step());
		STEP("read firing", FiredList fired = sim->readFiring());
		//! \note take care here: fired contains reference to internal buffers in sim.
		STEP("init global scatter", initGlobalScatter(fired.neurons, oreqs, obufs));
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

	//! \todo perhaps we should do a final waitGlobalScatter?
}


#undef STEP



void
Worker::initGlobalGather(req_vector& ireqs, fbuf_vector& ibufs)
{
	assert(mg_sourceNodes.size() == ibufs.size());
	assert(mg_sourceNodes.size() == ireqs.size());
	unsigned sid = 0;
	for(std::set<rank_t>::const_iterator source = mg_sourceNodes.begin();
			source != mg_sourceNodes.end(); ++source, ++sid) {
		MPI_LOG("Worker %u init gather from %u\n", m_rank, *source);
		assert(ibufs.find(*source) != ibufs.end());
		fbuf& incoming = ibufs[*source];
		incoming.clear();
		//! \todo do we do an incorrect copy operation here?
		ireqs.at(sid) = m_world.irecv(*source, WORKER_STEP, incoming);
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




/* Wait for all incoming firings sent during the previous cycle. Add these
 * firings to the queue */
void
Worker::waitGlobalGather(
		req_vector& ireqs,
		const fbuf_vector& ibufs,
		const nemo::ConnectivityMatrix& l_fcm,
		SpikeQueue& queue)
{
	using namespace boost::mpi;

	MPI_LOG("Worker %u waiting for messages from %lu peers\n", m_rank, ireqs.size());

#if 0 // see note below
	unsigned nreqs = ireqs.size();
	for(unsigned r=0; r < nreqs; ++r) {
		std::pair<status, req_vector::iterator> result = wait_any(ireqs.begin(), ireqs.end());
		rank_t sourceRank = result.first.source();
		assert(ibufs.find(sourceRank) != ibufs.end());
		const fbuf& incoming = ibufs.find(sourceRank)->second;
		MPI_LOG("Worker %u receiving %lu firings from %u\n", m_rank, incoming.size(), sourceRank);
		enqueueIncoming(incoming, l_fcm, queue);

	}
#endif

	/*! \note It should not be necessary to process these requests in order.
	 * However, the above commented-out code results in run-time errors, as it
	 * seems that some sources are received twice and others not at all. We
	 * should come back to this issue later.  */
	for(req_vector::iterator i = ireqs.begin(); i != ireqs.end(); ++i) {
		status result = i->wait();
		rank_t sourceRank = result.source();
		assert(ibufs.find(sourceRank) != ibufs.end());
		const fbuf& incoming = ibufs.find(sourceRank)->second;
		MPI_LOG("Worker %u receiving %lu firings from %u\n", m_rank, incoming.size(), sourceRank);
		enqueueIncoming(incoming, l_fcm, queue);
	}
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



/*
 * \param fired
 * 		Firing generated this cycle in the local simulation
 * \param oreqs
 * 		Per-/target/ rank buffer of send requests
 * \param obuf
 * 		Per-rank buffer of firing.
 */
void
Worker::initGlobalScatter(const fbuf& fired, req_vector& oreqs, fbuf_vector& obufs)
{
	MPI_LOG("Worker %u sending firing to %lu peers\n", m_rank, oreqs.size());

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

	unsigned tid = 0;
	for(std::set<rank_t>::const_iterator target = mg_targetNodes.begin();
			target != mg_targetNodes.end(); ++target, ++tid) {
		rank_t targetRank = *target;
		MPI_LOG("Worker %u sending %lu firings to %u\n", m_rank, obufs[targetRank].size(), targetRank);
		oreqs.at(tid) = m_world.isend(targetRank, WORKER_STEP, obufs[targetRank]);
	}
}



void
Worker::waitGlobalScatter(req_vector& oreqs)
{
	boost::mpi::wait_all(oreqs.begin(), oreqs.end());
}



	} // end namespace mpi
} // end namespace nemo
