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
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include "nemo_mpi_common.hpp"
#include "Mapper.hpp"
#include <NetworkImpl.hpp>


namespace nemo {
	namespace mpi {

void addNeuron(boost::mpi::communicator& world, nemo::NetworkImpl& net);


Worker::Worker(boost::mpi::communicator& world) :
	m_world(world)
{
	bool constructionDone = false;
	nemo::NetworkImpl net;
	int buf;

	while(!constructionDone) {
		boost::mpi::status msg = m_world.probe();
		switch(msg.tag()) {
			case NEURON_SCALAR: addNeuron(m_world, net); break;
			case SYNAPSE_VECTOR: addSynapseVector(net); break;
			case END_CONSTRUCTION: 
				world.recv(MASTER, END_CONSTRUCTION, buf);
				constructionDone = true;
				break;
			default:
				//! \todo deal with errors properly
				throw std::runtime_error("unknown tag");
		}
	}

	//! \todo now split net into local and global networks. Just strip net in-place.
	//! \todo exchange connectivity between nodes now

	m_world.barrier();
}


void
addNeuron(boost::mpi::communicator& world, nemo::NetworkImpl& net)
{
	std::pair<nidx_t, nemo::Neuron<float> > n;
	world.recv(MASTER, NEURON_SCALAR, n);
	net.addNeuron(n.first, n.second);
}



void
Worker::addSynapseVector(nemo::NetworkImpl& net)
{
	m_ss.clear();
	m_world.recv(MASTER, SYNAPSE_VECTOR, m_ss);
	for(std::vector<Synapse<unsigned, unsigned, float> >::const_iterator i = m_ss.begin();
			i != m_ss.end(); ++i) {
		const AxonTerminal<unsigned, float>& t = i->terminal;
		//! \todo add alternative addSynapse method which uses AxonTerminal directly
		net.addSynapse(i->source, t.target, i->delay, t.weight, t.plastic);
	}
}

	} // end namespace mpi
} // end namespace nemo
