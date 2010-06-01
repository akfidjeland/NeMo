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

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include "nemo_mpi_common.hpp"
#include "Mapper.hpp"
#include <NetworkImpl.hpp>


namespace nemo {
	namespace mpi {

void addNeuron(boost::mpi::communicator& world, nemo::NetworkImpl& net);


Worker::Worker(boost::mpi::communicator& world) :
	m_world(world),
	m_rank(world.rank()),
	ml_scount(0),
	mgi_scount(0),
	mgo_scount(0),
	m_ncount(0)
{
	int workers = m_world.size() - 1;
	Mapper mapper(workers);

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
				//! \todo deal with errors properly
				throw std::runtime_error("unknown tag");
		}
	}

	exchangeGlobalData(g_ss);

	m_world.barrier();

#if 0
	std::clog << "Worker " << m_rank << " " << m_ncount << " neurons" << std::endl;
	std::clog << "Worker " << m_rank << " " << ml_scount << " local synapses" << std::endl;
	std::clog << "Worker " << m_rank << " " << mgo_scount << " global synapses (outgoing)" << std::endl;
	std::clog << "Worker " << m_rank << " " << mgi_scount << " global synapses (incoming)" << std::endl;
#endif

	runSimulation();
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
			mg_fcm[svec.source].insert(i->target);
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
Worker::exchangeGlobalData(global_fcm_t& g_ss)
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
			ml_fcm.setRow(i->source, i->delay, i->terminals);
			mgi_scount += i->terminals.size();
		}
	}

	ml_fcm.finalize();
}



void
Worker::runSimulation()
{
	bool terminate = false;

	m_ireqs.resize(mg_sources.size());
	m_oreqs.resize(mg_targets.size());

	SimulationStep masterReq;

	initSendFiring();

	while(!masterReq.terminate) {
		m_mreq = m_world.irecv(MASTER, MASTER_STEP, masterReq);

		initReceiveFiring();
		//! \todo local gather
		boost::mpi::wait_all(m_oreqs.begin(), m_oreqs.end());
		/*! \todo Use wait any here instead, and accumulate input current as we get requests */
		boost::mpi::wait_all(m_ireqs.begin(), m_ireqs.end());
		m_mreq.wait();
		//! \todo local update
		initSendFiring();
		//! \todo local scatter
		/*! \todo Copy data back from simulation */
		std::clog << "Worker " << m_rank << " stepping\n";
	}
	std::clog << "Worker " << m_rank << " terminating\n";
}



void
Worker::initReceiveFiring()
{
	// dummy input
	std::vector<int> ibuf(mg_sources.size());
	unsigned sid = 0;
	for(std::set<rank_t>::const_iterator source = mg_sources.begin();
			source != mg_sources.end(); ++source, ++sid) {
		//std::clog << "Worker " << m_rank << " receiving sync from " << *source << std::endl;;
		//! \todo send actual data here
		m_ireqs[sid] = m_world.irecv(*source, WORKER_STEP, ibuf[sid]);
	}
}



void
Worker::initSendFiring()
{
	int obuf = 0;
	unsigned tid = 0;
	for(std::set<rank_t>::const_iterator target = mg_targets.begin();
			target != mg_targets.end(); ++target, ++tid) {
		//std::clog << "Worker " << m_rank << " sending sync to " << *target << std::endl;;
		m_oreqs[tid] = m_world.isend(*target, WORKER_STEP, obuf);
	}
}



	} // end namespace mpi
} // end namespace nemo
