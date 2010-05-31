/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Master.hpp"

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/utility.hpp>

#include <Network.hpp>
#include <NetworkImpl.hpp>
#include <ConfigurationImpl.hpp>
#include "nemo_mpi_common.hpp"
#include "Mapper.hpp"
#include <types.hpp>
#include <mpi_types.hpp>


namespace nemo {
	namespace mpi {

Master::Master(
		boost::mpi::communicator& world,
		const Network& net_,
		const Configuration& conf_) :
	m_world(world)
{
	distributeNetwork(net_.m_impl);

	/* The workers now exchange connectivity information. */

	m_world.barrier();

	/* We're now ready to run the simulation. The caller does this using class
	 * methods. */
}



Master::~Master()
{
	terminate();
}



unsigned
Master::workers() const
{
	return m_world.size() - 1;
}


//! \todo implement this in terms of iterators on some base class of //NetworkImpl.
void
Master::distributeNetwork(nemo::NetworkImpl* net)
{
	//! \todo base this on size hint as well
	unsigned wcount = workers();
	Mapper mapper(wcount);

	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net->m_neurons.begin();
			i != net->m_neurons.end(); ++i) {
		nidx_t source = i->first;
		int rank = mapper.rankOf(source);
		m_world.send(rank, NEURON_SCALAR, *i);
	}

	for(std::map<nidx_t, NetworkImpl::axon_t>::const_iterator axon = net->m_fcm.begin();
			axon != net->m_fcm.end(); ++axon) {

		nidx_t source = axon->first;
		int rank = mapper.rankOf(source);

		for(std::map<delay_t, NetworkImpl::bundle_t>::const_iterator bi = axon->second.begin();
				bi != axon->second.end(); ++bi) {
			delay_t delay = bi->first;
			const NetworkImpl::bundle_t& bundle = bi->second;
			//! \todo use a predefined output buffer
			SynapseVector svec(source, delay, bundle);
			m_world.send(rank, SYNAPSE_VECTOR, svec);
		}
	}

	for(int r=0; r < wcount; ++r) {
		m_world.send(r+1, END_CONSTRUCTION, int(0));
	}
}



void
Master::step(const std::vector<unsigned>& fstim)
{
	unsigned wcount = workers();
	//! \todo split up fstim here and insert into different requests
	SimulationStep data;
	std::vector<boost::mpi::request>reqs(wcount);
	for(int r=0; r < wcount; ++r) {
		reqs[r] = m_world.isend(r+1, SIM_STEP, data);
	}
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}



void
Master::terminate()
{
	unsigned wcount = workers();
	SimulationStep data(true, std::vector<unsigned>());
	std::vector<boost::mpi::request>reqs(wcount);
	for(int r=0; r < wcount; ++r) {
		reqs[r] = m_world.isend(r+1, SIM_STEP, data);
	}
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}



	} // end namespace mpi
} // end namespace nemo
