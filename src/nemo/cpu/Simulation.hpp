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

#include <vector>

#include <nemo/config.h>
#include <nemo/internal_types.h>
#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/FiringBuffer.hpp>
#include <nemo/Neurons.hpp>
#include <nemo/RandomMapper.hpp>
#include <nemo/Timer.hpp>

#include "Worker.hpp"
#include "Neurons.hpp"


namespace nemo {
	namespace cpu {

class NEMO_CPU_DLL_PUBLIC Simulation : public nemo::SimulationBackend
{
	public:

		Simulation(const network::Generator&, const nemo::ConfigurationImpl&);

		unsigned getFractionalBits() const;

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::initCurrentStimulus */
		void initCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::addCurrentStimulus */
		void addCurrentStimulus(nidx_t neuron, float current);

		/*! \copydoc nemo::SimulationBackend::finalizeCurrentStimulus */
		void finalizeCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::prefire */
		void prefire() { }

		/*! \copydoc nemo::SimulationBackend::fire */
		void fire();

		/*! \copydoc nemo::SimulationBackend::postfire */
		void postfire() { }

		/*! \copydoc nemo::SimulationBackend::readFiring */
		FiredList readFiring();

		/*! \copydoc nemo::SimulationBackend::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::SimulationBackend::setNeuron */
		void setNeuron(unsigned idx, const float param[], const float state[]);

		/*! \copydoc nemo::Simulation::setNeuronState */
		void setNeuronState(unsigned neuron, unsigned var, float val);

		/*! \copydoc nemo::Simulation::setNeuronParameter */
		void setNeuronParameter(unsigned neuron, unsigned parameter, float val);

		/*! \copydoc nemo::Simulation::getNeuronState */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*! \copydoc nemo::Simulation::getNeuronParameter */
		float getNeuronParameter(unsigned neuron, unsigned param) const;

		/*! \copydoc nemo::Simulation::getMembranePotential */
		float getMembranePotential(unsigned neuron) const;

		/*! \copydoc nemo::Simulation::getSynapsesFrom */
		const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

		/*! \copydoc nemo::Simulation::getSynapseTarget */
		unsigned getSynapseTarget(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapseDelay */
		unsigned getSynapseDelay(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapseWeight */
		float getSynapseWeight(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapsePlastic */
		unsigned char getSynapsePlastic(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

	private:

		Neurons m_neurons;

		const RandomMapper<nidx_t>& m_mapper;

		typedef std::vector<fix_t> current_vector_t;

		//! \todo can we get rid of this?
		size_t m_neuronCount;

		/* last cycles firing, one entry per neuron */
		std::vector<unsigned> m_fired;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* bit-mask containing delays at which neuron has *any* outoing
		 * synapses */
		std::vector<uint64_t> m_delays;

		/*! Update state of all neurons */
		void update(const current_vector_t&);

		nemo::ConnectivityMatrix m_cm;

		/* accumulated current from incoming spikes for each neuron */
		std::vector<fix_t> m_current;

		/*! Deliver spikes due for delivery */
		current_vector_t& deliverSpikes();

		void setFiring();

		FiringBuffer m_firingBuffer;

#ifdef NEMO_CPU_MULTITHREADED

		std::vector<Worker> m_workers;

		void initWorkers(size_t neurons, unsigned threads);

		friend class Worker;
#endif

		void updateRange(int begin, int end);

		void deliverSpikesOne(nidx_t source, delay_t delay);

		Timer m_timer;

		nidx_t validLocalIndex(unsigned g_idx) const;
};



/* If threadCount is -1, use default values */
NEMO_CPU_DLL_PUBLIC
void chooseHardwareConfiguration(nemo::ConfigurationImpl&, int threadCount = -1);

	} // namespace cpu
} // namespace nemo


#endif
