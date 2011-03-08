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

#include <nemo/Network.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/Configuration.hpp>
#include <nemo/ConfigurationImpl.hpp>
#include "nemo_mpi_common.hpp"
#include <nemo/types.hpp>

#include "types.hpp"
#include "log.hpp"


namespace nemo {
	namespace mpi {

Master::Master(
		boost::mpi::environment& env,
		boost::mpi::communicator& world,
		const Network& net,
		const Configuration& conf) :
	m_world(world),
	m_mapper(net.neuronCount(), m_world.size() - 1, m_world.rank())
{
	MPI_LOG("Master starting on %s\n", env.processor_name().c_str());

	/* Need a dummy entry, to pop on first call to readFiring */
	m_firing.push_back(std::vector<unsigned>());

	/* send configuration from master to all workers */
	boost::mpi::broadcast(world, *conf.m_impl, MASTER);

	/* send overall network size to all workers */
	unsigned neurons = net.neuronCount();
	boost::mpi::broadcast(world, neurons, MASTER);

	distributeNeurons(m_mapper, *net.m_impl);
	distributeSynapses(m_mapper, *net.m_impl);

	/* The workers now set up the local simulations. This could take some time. */

	m_world.barrier();

	/* We're now ready to run the simulation. The caller does this using class
	 * methods. */

	m_timer.reset();
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.reset();
#endif
}



Master::~Master()
{
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.report(0);
#endif
	terminate();
}



unsigned
Master::workers() const
{
	return m_world.size() - 1;
}



template<class T>
void
flushBuffer(int tag,
		std::vector<T>& input,
		std::vector< std::vector<T> >& output,
		boost::mpi::communicator& world)
{
	broadcast(world, tag, MASTER);
	scatter(world, output, input, MASTER);
	for(typename std::vector< std::vector<T> >::iterator i = output.begin(); i != output.end(); ++i) {
		i->clear();
	}
}


void
Master::distributeNeurons(const Mapper& mapper, const network::Generator& net)
{
	typedef std::vector< std::pair<nidx_t, Neuron<float> > > nvector;
	nvector input; // dummy
	std::vector<nvector> output(m_world.size());
	unsigned queued = 0;
	//! \todo pass this in
	const unsigned bufferSize = 2 << 11;

	for(network::neuron_iterator n = net.neuron_begin(); n != net.neuron_end(); ++n, ++queued) {
		output.at(mapper.rankOf(n->first)).push_back(*n);
		if(queued == bufferSize) {
			flushBuffer(NEURON_VECTOR, input, output, m_world);
			queued = 0;
		}
	}

	flushBuffer(NEURON_VECTOR, input, output, m_world);
	int tag = NEURONS_END;
	broadcast(m_world, tag, MASTER);
}



void
Master::distributeSynapses(const Mapper& mapper, const network::Generator& net)
{
	typedef std::vector<Synapse> svector;
	svector input; // dummy
	std::vector<svector> output(m_world.size());
	unsigned queued = 0;

	//! \todo pass this in
	const unsigned bufferSize = 2 << 11;

	for(network::synapse_iterator s = net.synapse_begin(); s != net.synapse_end(); ++s, ++queued) {
		int sourceRank = mapper.rankOf(s->source);
		int targetRank = mapper.rankOf(s->target());
		output.at(sourceRank).push_back(*s);
		if(sourceRank != targetRank) {
			output.at(targetRank).push_back(*s);
		}
		if(queued == bufferSize) {
			flushBuffer(SYNAPSE_VECTOR, input, output, m_world);
			queued = 0;
		}
	}
	flushBuffer(SYNAPSE_VECTOR, input, output, m_world);
	int tag = SYNAPSES_END;
	broadcast(m_world, tag, MASTER);
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
	m_timer.step();

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.reset();
#endif

	unsigned wcount = workers();

	std::vector<SimulationStep> oreqData(wcount);
	std::vector<boost::mpi::request> oreqs(wcount);

	distributeFiringStimulus(m_mapper, fstim, oreqData);
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	for(unsigned r=0; r < wcount; ++r) {
		oreqs.at(r) = m_world.isend(r+1, MASTER_STEP, oreqData.at(r));
	}
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	boost::mpi::wait_all(oreqs.begin(), oreqs.end());
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	m_firing.push_back(std::vector<unsigned>());

	std::vector<unsigned> dummy_fired;
	std::vector< std::vector<unsigned> > fired;
	gather(m_world, dummy_fired, fired, MASTER);

	/* If neurons are allocated to nodes ordered by neuron index, we can get a
	 * sorted list by just concatening the per-node lists in rank order */
	for(unsigned r=0; r < wcount; ++r) {
		const std::vector<unsigned>& node_fired = fired.at(r+1);
		std::copy(node_fired.begin(), node_fired.end(), std::back_inserter(m_firing.back()));
#ifdef NEMO_MPI_DEBUG_TRACE
		MPI_LOG("Master received %u firings from %u\n", node_fired.size(), r+1);
#endif
	}

#ifdef NEMO_MPI_DEBUG_TRACE
	std::copy(m_firing.back().begin(), m_firing.back().end(), std::ostream_iterator<unsigned>(std::cout, " "));
#endif
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
	m_mpiTimer.step();
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



unsigned long
Master::elapsedWallclock() const
{
	return m_timer.elapsedWallclock();
}



unsigned long
Master::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}


void
Master::resetTimer()
{
	m_timer.reset();
}


	} // end namespace mpi
} // end namespace nemo
