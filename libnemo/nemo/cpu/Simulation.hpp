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

#include <nemo/config.h>
#include <nemo/types.h>
#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/STDP.hpp>
#include <nemo/Timer.hpp>
#include <nemo/RNG.hpp>

#include "Worker.hpp"


namespace nemo {
	namespace cpu {

class Simulation : public nemo::SimulationBackend
{
	public:

		Simulation(const nemo::NetworkImpl& net, const nemo::ConfigurationImpl& conf);

		unsigned getFractionalBits() const;

		// there's no real limit here, but return something anyway
		/*! \copydoc nemo::Simulation::getFiringBufferLength */
		unsigned getFiringBufferLength() const { return 10000; }

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::step */
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
		std::vector<nemo::RNG> m_rng;

		/* Accumulated firing history since last flush */
		unsigned m_lastFlush;
		std::vector<unsigned int> m_firedCycle;
		std::vector<unsigned int> m_firedNeuron;
		/* externally exposed copy of the same data */
		std::vector<unsigned int> m_firedCycleExt;
		std::vector<unsigned int> m_firedNeuronExt;
		void setFiring();

#ifdef NEMO_CPU_MULTITHREADED

		//! \todo allow user to determine number of threads.
		static const int m_nthreads = 4;
		//! \todo use a std vector here
		Worker* m_workers[m_nthreads];

		void initThreads(size_t ncount);

		friend class Worker;
#endif

		void updateRange(int begin, int end);

		void deliverSpikesOne(nidx_t source, delay_t delay);

		void accumulateStdp();

		STDP<float> m_stdp;

		weight_t updateRegion(uint64_t spikes, nidx_t source, nidx_t target);

		Timer m_timer;
};


	} // namespace cpu
} // namespace nemo


#endif
