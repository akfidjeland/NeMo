/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Master.hpp"

#include <iterator>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/utility.hpp>

#include <nemo_config.h>
#include <Network.hpp>
#include <NetworkImpl.hpp>
#include <ConfigurationImpl.hpp>
#include "nemo_mpi_common.hpp"
#include <types.hpp>
#include <mpi_types.hpp>


namespace nemo {
	namespace mpi {

Master::Master(
		boost::mpi::communicator& world,
		const Network& net_,
		const Configuration& conf_) :
	m_world(world),
	//! \todo base this on size hint as well
	m_mapper(m_world.size() - 1, m_world.rank())
{
	/* Need a dummy entry, to pop on first call to readFiring */
	m_firing.push_back(std::vector<unsigned>());

	distributeNetwork(m_mapper, net_.m_impl);

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
Master::distributeNetwork(
		const Mapper& mapper,
		const nemo::NetworkImpl* net)
{
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

	unsigned wcount = workers();
	for(unsigned r=0; r < wcount; ++r) {
		m_world.send(r+1, END_CONSTRUCTION, int(0));
	}
}


void
distributeFiringStimulus(
		const Mapper& mapper,
		const std::vector<unsigned>& fstim,
		std::vector<SimulationStep>& reqs)
{
	for(std::vector<unsigned>::const_iterator i = fstim.begin();
			i != fstim.end(); ++i) {
		nidx_t neuron = nidx_t(*i);
		assert(unsigned(mapper.rankOf(neuron) - 1) < reqs.size());
		reqs.at(mapper.rankOf(neuron) - 1).forceFiring(neuron);
	}
}



void
Master::step(const std::vector<unsigned>& fstim)
{
	unsigned wcount = workers();

	std::vector<SimulationStep> oreqData(wcount);
	std::vector<boost::mpi::request> oreqs(wcount);

	distributeFiringStimulus(m_mapper, fstim, oreqData);

	for(unsigned r=0; r < wcount; ++r) {
		oreqs.at(r) = m_world.isend(r+1, MASTER_STEP, oreqData.at(r));
	}

	boost::mpi::wait_all(oreqs.begin(), oreqs.end());

	/* Ideally we'd get the results back in order. Just insert in rank order to
	 * avoid having to sort later */
	std::vector<unsigned> ibuf;
	m_firing.push_back(std::vector<unsigned>());

	for(unsigned r=0; r < wcount; ++r) {
		m_world.recv(r+1, MASTER_STEP, ibuf);
		std::copy(ibuf.begin(), ibuf.end(), std::back_inserter(m_firing.back()));
#ifdef MPI_LOGGING
		std::copy(ibuf.begin(), ibuf.end(),
				std::ostream_iterator<unsigned>(std::cout, " "));
#endif
	}
#ifdef MPI_LOGGING
	if(ibuf.size() > 0) {
		std::cout << std::endl;
	}
#endif
}



void
Master::terminate()
{
	unsigned wcount = workers();
	SimulationStep data(true, std::vector<unsigned>());
	std::vector<boost::mpi::request>reqs(wcount);
	for(unsigned r=0; r < wcount; ++r) {
		reqs[r] = m_world.isend(r+1, MASTER_STEP, data);
	}
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}



const std::vector<unsigned>&
Master::readFiring()
{
	//! \todo deal with underflow here
	m_firing.pop_front();
	return m_firing.front();
}



	} // end namespace mpi
} // end namespace nemo
