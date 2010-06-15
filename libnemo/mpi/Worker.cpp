/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Worker.hpp"

#include <stdexcept>
#include <algorithm>

#include <boost/scoped_ptr.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include <nemo.hpp>
#include <NetworkImpl.hpp>
#include <ConfigurationImpl.hpp>

#include "Mapper.hpp"
#include "SpikeQueue.hpp"
#include "nemo_mpi_common.hpp"
#include "log.hpp"


namespace nemo {
	namespace mpi {


Worker::Worker(
		boost::mpi::environment& env,
		boost::mpi::communicator& world) :
	m_world(world),
	m_rank(world.rank()),
	ml_scount(0),
	mgi_scount(0),
	mgo_scount(0),
	m_ncount(0)
{
	MPI_LOG("Starting worker %u on %s\n", world.rank(),
			env.processor_name().c_str());

	nemo::Configuration conf;
	boost::mpi::broadcast(m_world, *conf.m_impl, MASTER);
	conf.disableLogging();

	unsigned neurons;
	boost::mpi::broadcast(m_world, neurons, MASTER);

	int workers = m_world.size() - 1;
	Mapper mapper(neurons, workers, m_rank);

	bool constructionDone = false;
	nemo::NetworkImpl net;
	int buf;

	global_fcm_t g_ss;

	while(!constructionDone) {
		boost::mpi::status msg = m_world.probe();
		switch(msg.tag()) {
			case NEURON_SCALAR: addNeuron(net); break;
			case SYNAPSE_VECTOR: addSynapseVector(mapper, net, g_ss); break;
			case END_CONSTRUCTION: 
				world.recv(MASTER, END_CONSTRUCTION, buf);
				constructionDone = true;
				break;
			default:
				//! \todo throw mpi error here instead. Remove stdexcept header when doing so
				//! \todo deal with errors properly
				throw std::runtime_error("unknown tag");
		}
	}

	exchangeGlobalData(mapper, g_ss);

	m_world.barrier();

	MPI_LOG("Worker %u: %u neurons\n", m_rank, m_ncount);
	MPI_LOG("Worker %u: %u local synapses\n", m_rank, ml_scount);
	MPI_LOG("Worker %u: %u global synapses (out)\n", m_rank,  mgo_scount);
	MPI_LOG("Worker %u: %u global synapses (int)\n", m_rank, mgi_scount);

	runSimulation(net, conf, mapper.localCount());
}


void
Worker::addNeuron(nemo::NetworkImpl& net)
{
	std::pair<nidx_t, nemo::Neuron<float> > n;
	m_world.recv(MASTER, NEURON_SCALAR, n);
	net.addNeuron(n.first, n.second);
	m_ncount++;
}



void
Worker::addSynapseVector(const Mapper& mapper,
		nemo::NetworkImpl& net,
		global_fcm_t& g_ss)
{

	/* Incoming data from master */
	//! \todo allocate this only once
	SynapseVector svec;
	m_world.recv(MASTER, SYNAPSE_VECTOR, svec);

	typedef std::map<rank_t, std::vector<SynapseVector::terminal_t> > acc_t;
	acc_t ss;

	// now, for each synapse determine where it should go: local or global
	for(std::vector<SynapseVector::terminal_t>::const_iterator i = svec.terminals.begin();
			i != svec.terminals.end(); ++i) {
		int targetRank = mapper.rankOf(i->target);
		if(targetRank == m_rank) {
			//! \todo add alternative addSynapse method which uses AxonTerminal directly
			net.addSynapse(svec.source, i->target, svec.delay, i->weight, i->plastic);
			ml_scount++;
		} else {
			mg_fcm[svec.source].insert(targetRank);
			mgo_scount++;
			ss[targetRank].push_back(*i);
		}
	}

	for(acc_t::const_iterator i = ss.begin(); i != ss.end(); ++i) {
		rank_t targetRank = i->first;
		g_ss[targetRank].push_back(SynapseVector(svec.source, svec.delay, i->second));
	}
}



void
Worker::exchangeGlobalData(const Mapper& mapper, global_fcm_t& g_ss)
{
	for(rank_t targetOffset = 1; targetOffset < m_world.size() - 1; ++targetOffset) {
		/* We need to send something to all targets, so just use default-
		 * constructed entry if nothing is present */
		rank_t source = 1 + ((m_rank - 1 + (m_world.size() - 1) - targetOffset) % (m_world.size() - 1));
		rank_t target = 1 + ((m_rank - 1 + targetOffset) % (m_world.size() - 1));
		boost::mpi::request reqs[2];

		m_obuf = g_ss[target];
		if(m_obuf.size() != 0) {
			mg_targets.insert(target);
		}
		reqs[0] = m_world.isend(target, SYNAPSE_VECTOR, m_obuf);
		g_ss.erase(target);

		//! \todo probably not needed
		m_ibuf.clear();
		reqs[1] = m_world.irecv(source, SYNAPSE_VECTOR, m_ibuf);
		boost::mpi::wait_all(reqs, reqs+2);
		if(m_ibuf.size() != 0) {
			mg_sources.insert(source);
		}

		for(std::vector<SynapseVector>::const_iterator i = m_ibuf.begin();
				i != m_ibuf.end(); ++i) {
			Row& row = ml_fcm.setRow(i->source, i->delay, i->terminals);

			/* The source sends synapses with global target indices. At
			 * run-time we need local addresses instead */
			for(size_t s=0; s < row.len; ++s) {
				row.data[s].target = mapper.localIndex(row.data[s].target);
			}
			mgi_scount += i->terminals.size();
		}
	}

	ml_fcm.finalize();
}


void
gather(const SpikeQueue& queue,
		const ConnectivityMatrix& fcm,
		std::vector<weight_t>& current)
{
	std::fill(current.begin(), current.end(), 0.0);

	SpikeQueue::const_iterator arrival_end = queue.current_end();
	for(SpikeQueue::const_iterator arrival = queue.current_begin();
			arrival != arrival_end; ++arrival) {
		const Row& row = fcm.getRow(arrival->source(), arrival->delay());
		for(unsigned s=0; s < row.len; ++s) {
			const FAxonTerminal& terminal = row.data[s];
			assert(terminal.target < current.size());
			current.at(terminal.target) += terminal.weight;
		}
	}
}


void
Worker::runSimulation(const nemo::NetworkImpl& net,
		const nemo::Configuration& conf,
		size_t localCount)
{
	/* Local simulation data */
	boost::scoped_ptr<nemo::Simulation> sim(nemo::Simulation::create(net, conf));
	const std::vector<unsigned>* l_firedCycles;         // unused
	const std::vector<unsigned>* l_fired;               // neurons
	std::vector<weight_t> istim(localCount, 0);         // input from global spikes
	SpikeQueue queue(net.maxDelay());

	/* Incoming master request */
	boost::mpi::request mreq;
	SimulationStep masterReq;

	/* Incoming peer requests */
	req_vector ireqs(mg_sources.size());
	fbuf_vector ibufs(mg_sources.size());

	/* Outgoing peer requests. Not all peers are necessarily potential targets
	 * for local neurons, so the output buffer could in principle be smaller.
	 * Allocated it with potentially unused entries so that insertion (which is
	 * frequent) is faster */
	req_vector oreqs(mg_targets.size());
	rank_t maxTargetRank = *std::max_element(mg_targets.begin(), mg_targets.end());
	assert(1 + maxTargetRank >= rank_t(mg_targets.size()));
	fbuf_vector obufs(1 + maxTargetRank);

	/* Scatter empty firing packages to start with */
	initGlobalScatter(fbuf(), oreqs, obufs);

	while(!masterReq.terminate) {
		mreq = m_world.irecv(MASTER, MASTER_STEP, masterReq);

		initGlobalGather(ireqs, ibufs);
		//! \todo local gather
		waitGlobalScatter(oreqs);
		waitGlobalGather(ireqs, ibufs, queue);
		mreq.wait();
		//! \todo improve naming
		gather(queue, ml_fcm, istim);
		//! \todo split up step and only do neuron update here
		sim->step(masterReq.fstim, istim);
		sim->readFiring(&l_firedCycles, &l_fired);
		initGlobalScatter(*l_fired, oreqs, obufs);
		sendMaster(*l_fired);
		queue.step();
		//! \todo local scatter
	}

	//! \todo perhaps we should do a final waitGlobalScatter?
}



void
Worker::initGlobalGather(req_vector& ireqs, fbuf_vector& ibufs)
{
	unsigned sid = 0;
	for(std::set<rank_t>::const_iterator source = mg_sources.begin();
			source != mg_sources.end(); ++source, ++sid) {
		//! \todo need to make sure that every entry here is set.
		ireqs.at(sid) = m_world.irecv(*source, WORKER_STEP, ibufs.at(sid));
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
Worker::waitGlobalGather(
		req_vector& ireqs,
		const fbuf_vector& ibufs,
		SpikeQueue& queue)
{
	using namespace boost::mpi;

	std::pair<status, std::vector<request>::iterator> result;

	unsigned nreqs = ireqs.size();

	for(unsigned r=0; r < nreqs; ++r) {
		result = wait_any(ireqs.begin(), ireqs.end());
#ifdef MPI_LOGGING
		const status& incoming = result.first;
		std::cerr << "Worker " << m_rank
			<< " receiving " << ibufs.at(r).size() << " firings from "
			<< incoming.source() << std::endl;
#endif
		enqueueIncoming(ibufs.at(r), ml_fcm, queue);
	}
}



/*
 * \param fired
 * 		Firing generated this cycle in the local simulation
 * \param oreqs
 * 		Per-/target/ rank buffer of send requests
 * \param obuf
 * 		Per-rank buffer of firing.
 *
 * \pre obufs are all empty
 * \post obufs are all empty
 */
void
Worker::initGlobalScatter(
		const fbuf& fired,
		req_vector& oreqs,
		fbuf_vector& obufs)
{
	/* Each local firing may be sent to zero or more peers */
	for(std::vector<unsigned>::const_iterator source = fired.begin();
			source != fired.end(); ++source) {
		const std::set<rank_t>& targets = mg_fcm[*source];
		for(std::set<rank_t>::const_iterator target = targets.begin();
				target != targets.end(); ++target) {
			rank_t targetRank = *target;
			assert(targetRank < rank_t(obufs.size()));
			assert(mg_targets.count(targetRank) == 1);
			obufs.at(targetRank).push_back(*source);
		}
	}

	unsigned tid = 0;
	for(std::set<rank_t>::const_iterator target = mg_targets.begin();
			target != mg_targets.end(); ++target, ++tid) {
		rank_t targetRank = *target;
#ifdef MPI_LOGGING
		std::cerr << "Worker " << m_rank << " sending " << obufs.at(targetRank).size()
			<< " firings to " << *target << std::endl;
#endif
		oreqs.at(tid) = m_world.isend(targetRank, WORKER_STEP, obufs.at(targetRank));
		obufs.at(targetRank).clear();
	}
}



void
Worker::waitGlobalScatter(req_vector& oreqs)
{
	boost::mpi::wait_all(oreqs.begin(), oreqs.end());
}



/* Send most recent cycle's firing back to master */
void
Worker::sendMaster(const fbuf& fired)
{
	//! \todo async send here?
	m_world.send(MASTER, MASTER_STEP, fired);
}


	} // end namespace mpi
} // end namespace nemo
