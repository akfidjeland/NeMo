#ifndef NEMO_CUDA_SIMULATION_HPP
#define NEMO_CUDA_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <boost/optional.hpp>

#include <nemo/config.h>
#include <nemo/STDP.hpp>
#include <nemo/Timer.hpp>
#include <nemo/internal_types.h>
#include <nemo/ConfigurationImpl.hpp>
#include <nemo/SimulationBackend.hpp>

#include "Mapper.hpp"
#include "NVector.hpp"
#include "ConnectivityMatrix.hpp"
#include "CycleCounters.hpp"
#include "DeviceAssertions.hpp"
#include "FiringBuffer.hpp"
#include "NeuronParameters.hpp"
#include "ThalamicInput.hpp"

namespace nemo {

	namespace network {
		class Generator;
	}

	namespace cuda {


class Simulation : public nemo::SimulationBackend
{
	public :

		~Simulation();

		/* CONFIGURATION */

		unsigned getFractionalBits() const;

		/* SIMULATION */

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& nidx);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::initCurrentStimulus */
		void initCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::addCurrentStimulus */
		void addCurrentStimulus(nidx_t neuron, float current);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void finalizeCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::update */
		void update();

		/*! \copydoc nemo::SimulationBackend::readFiring */
		FiredList readFiring();

		/*! \copydoc nemo::Simulation::getMembranePotential */
		float getMembranePotential(unsigned neuron) const;

		/*! \copydoc nemo::Simulation::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::Simulation::getTargets */
		const std::vector<unsigned>& getTargets(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getDelays */
		const std::vector<unsigned>& getDelays(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getWeights */
		const std::vector<float>& getWeights(const std::vector<synapse_id>&);

		/*! \copydoc nemo::Simulation::getPlastic */
		const std::vector<unsigned char>& getPlastic(const std::vector<synapse_id>&);

		void finishSimulation();

		/* TIMING */

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

		/*! \copydoc nemo::SimulationBackend::mapper */
		virtual Mapper& mapper() { return m_mapper; }

	private :

		/* Use factory method for generating objects */
		Simulation(const network::Generator&, const nemo::ConfigurationImpl&);

		friend SimulationBackend* simulation(const network::Generator& net, const ConfigurationImpl& conf);

		Mapper m_mapper;

		nemo::ConfigurationImpl m_conf;

		//! \todo add this to logging output
		/*! \return
		 * 		number of bytes allocated on the device
		 *
		 * It seems that cudaMalloc*** does not fail properly when running out
		 * of memory, so this value could be useful for diagnostic purposes */
		size_t d_allocated() const;

		NeuronParameters m_neurons;

		ConnectivityMatrix m_cm;

		NVector<uint64_t, 2> m_recentFiring;

		ThalamicInput m_thalamicInput;

		/* Densely packed, one bit per neuron */
		NVector<uint32_t> m_firingStimulus;
		void clearFiringStimulus();

		NVector<fix_t> m_currentStimulus;

		/* The firing buffer keeps data for a certain duration. One bit is
		 * required per neuron (regardless of whether or not it's firing */
		FiringBuffer m_firingBuffer;

		CycleCounters m_cycleCounters;

		DeviceAssertions m_deviceAssertions;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		boost::optional<StdpFunction> m_stdp;

		void configureStdp();

		Timer m_timer;

		/* Device pointers to simulation stimulus. The stimulus may be set
		 * separately from the step, hence member variables */
		uint32_t* md_fstim;
		fix_t* md_istim;
};

	} // end namespace cuda
} // end namespace nemo

#endif
