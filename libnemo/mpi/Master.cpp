/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Master.hpp"

#include <boost/mpi/environment.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/serialization/utility.hpp>

#include <Network.hpp>
#include <NetworkImpl.hpp>
#include <ConfigurationImpl.hpp>
#include "nemo_mpi_common.hpp"
#include "Mapper.hpp"
#include <types.hpp>


namespace nemo {
	namespace mpi {

Master::Master(
		boost::mpi::communicator& world,
		const Network& net_,
		const Configuration& conf_) :
	m_world(world)
{
	distributeNetwork(net_.m_impl);

	/* The workers now exchange connectivity information. When
	 * that's done we're ready to start simulation. */
	m_world.barrier();
}



//! \todo implement this in terms of iterators on some base class of //NetworkImpl.
void
Master::distributeNetwork(nemo::NetworkImpl* net)
{
	int workers = m_world.size() - 1;
	//! \todo base this on size hint as well
	Mapper mapper(workers);

	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net->m_neurons.begin();
			i != net->m_neurons.end(); ++i) {
		nidx_t source = i->first;
		int rank = mapper.rankOf(source);
		m_world.send(rank, NEURON_SCALAR, *i);
	}

	for(std::map<nidx_t, NetworkImpl::axon_t>::const_iterator axon = net->m_fcm.begin();
			axon != net->m_fcm.end(); ++axon) {

		m_ss.clear();
		nidx_t source = axon->first;
		int rank = mapper.rankOf(source);

		for(std::map<delay_t, NetworkImpl::bundle_t>::const_iterator bi = axon->second.begin();
				bi != axon->second.end(); ++bi) {

			delay_t delay = bi->first;
			NetworkImpl::bundle_t bundle = bi->second;

			for(NetworkImpl::bundle_t::const_iterator si = bundle.begin();
					si != bundle.end(); ++si) {
				m_ss.push_back(Synapse<unsigned, unsigned, float>(source, delay, *si));
			}
		}

		m_world.send(rank, SYNAPSE_VECTOR, m_ss);
	}

	for(int r=0; r < workers; ++r) {
		m_world.send(r+1, END_CONSTRUCTION, int(0));
	}
}



	} // end namespace mpi
} // end namespace nemo
