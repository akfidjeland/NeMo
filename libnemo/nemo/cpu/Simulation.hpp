#ifndef NEMO_CPU_SIMULATION_HPP
#define NEMO_CPU_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <vector>
#ifdef PTHREADS_ENABLED
#include <pthread.h>
#endif

#include <nemo/types.h>
#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/STDP.hpp>
#include <nemo/Timer.hpp>

#include "RNG.hpp"


namespace nemo {

	class Network;

	namespace cpu {

#ifdef PTHREADS_ENABLED

struct Job {

	Job(): start(0), end(0), fstim(NULL), sim(NULL) {}

	Job(size_t start, size_t end, size_t ncount, struct Simulation* sim) :
		start(start), end(end), fstim(NULL), sim(sim) {
	}

	size_t start;
	size_t end;

	// input - full vector
	unsigned int* fstim;

	struct Simulation* sim;

} __attribute((aligned(ASSUMED_CACHE_LINE_SIZE)));

#endif


class Simulation : public nemo::SimulationBackend
{
	public:

		Simulation(const nemo::NetworkImpl& net, const nemo::ConfigurationImpl& conf);

		~Simulation();

		unsigned getFractionalBits() const;

		// there's no real limit here, but return something anyway
		/*! \copydoc nemo::Simulation::getFiringBufferLength */
		unsigned getFiringBufferLength() const { return 10000; }

		//! \todo implement getFractionalBits

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::step */
		//! \todo tidy!
		//void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());
		void step();

		/*! \copydoc nemo::SimulationBackend::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::Simulation::readFiring */
		unsigned readFiring(
				const std::vector<unsigned>** cycles,
				const std::vector<unsigned>** nidx);

		/*! \copydoc nemo::Simulation::flushFiringBuffer */
		void flushFiringBuffer();

		/*! \copydoc nemo::Simulation::getSynapses */
		void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic);

		/*! \copydoc nemo::SimulationBackend::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::SimulationBackend::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::SimulationBackend::resetTimer */
		void resetTimer();


	private:

		typedef std::vector<fix_t> current_vector_t;
		typedef std::vector<unsigned> stimulus_vector_t;

		size_t m_neuronCount;

		/* At run-time data is put into regular vectors for vectorizable
		 * operations */
		//! \todo enforce 16-byte allignment to support vectorisation
		std::vector<float> m_a;
		std::vector<float> m_b;
		std::vector<float> m_c;
		std::vector<float> m_d;

		std::vector<float> m_u;
		std::vector<float> m_v;
		std::vector<float> m_sigma;

		/* last cycles firing, one entry per neuron */
		std::vector<unsigned> m_fired;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* Set all neuron parameters from input network in local data structures */
		void setNeuronParameters(const nemo::NetworkImpl& net);

		/*! Update state of all neurons */
		void update(const stimulus_vector_t&, const current_vector_t&);

		void setConnectivityMatrix(const nemo::NetworkImpl& net);

		nemo::ConnectivityMatrix m_cm;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fix_t> m_current;

		/*! Deliver spikes due for delivery */
		current_vector_t& deliverSpikes();

		/* firing stimulus (for a single cycle) */
		stimulus_vector_t m_fstim;

		//! \todo may want to have one rng per neuron or at least per thread
		std::vector<RNG> m_rng;

		void initialiseRng(const nemo::NetworkImpl& net);

		/* Accumulated firing history since last flush */
		std::vector<unsigned int> m_firedCycle;
		std::vector<unsigned int> m_firedNeuron;
		/* externally exposed copy of the same data */
		std::vector<unsigned int> m_firedCycleExt;
		std::vector<unsigned int> m_firedNeuronExt;
		void setFiring();

#ifdef PTHREADS_ENABLED
		//! \todo allow user to determine number of threads
		static const int m_nthreads = 4;
		pthread_t m_thread[m_nthreads];
		pthread_attr_t m_thread_attr[m_nthreads];
		Job* m_job[m_nthreads];

		void initThreads(size_t ncount);

		friend void* start_thread(void*);
#endif
		uint m_cycle;

		void updateRange(int begin, int end, const unsigned int fstim[]);

		void deliverSpikesOne(nidx_t source, delay_t delay);

		void accumulateStdp();

		STDP<float> m_stdp;

		weight_t updateRegion(uint64_t spikes, nidx_t source, nidx_t target);

		Timer m_timer;
};


	} // namespace cpu
} // namespace nemo


#endif
