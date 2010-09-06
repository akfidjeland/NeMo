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
#include "FiringOutput.hpp"
#include "NeuronParameters.hpp"
#include "ThalamicInput.hpp"

namespace nemo {

	class NetworkImpl;

	namespace cuda {


class Simulation : public nemo::SimulationBackend
{
	public :

		~Simulation();

		/* CONFIGURATION */

		unsigned getFractionalBits() const;

		/* NETWORK SIMULATION */

		void setFiringStimulus(const std::vector<unsigned>& nidx);

		void setCurrentStimulus(const std::vector<fix_t>& current);

		void step();

		void applyStdp(float reward);

		void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic);

		unsigned readFiring(const std::vector<unsigned>** cycles, const std::vector<unsigned>** nidx);

		void flushFiringBuffer();

		void finishSimulation();

		/* TIMING */
		unsigned long elapsedWallclock() const;
		unsigned long elapsedSimulation() const;
		void resetTimer();

	private :

		/* Use factory method for generating objects */
		Simulation(const nemo::NetworkImpl&, const nemo::ConfigurationImpl&);

		friend SimulationBackend* simulation(const NetworkImpl& net, const ConfigurationImpl& conf);

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

		NVector<uint64_t> m_recentFiring;

		ThalamicInput m_thalamicInput;

		/* Densely packed, one bit per neuron */
		NVector<uint32_t> m_firingStimulus;
		void clearFiringStimulus();

		NVector<fix_t> m_currentStimulus;
		void clearCurrentStimulus();

		/* The firing buffer keeps data for a certain duration. One bit is
		 * required per neuron (regardless of whether or not it's firing */
		FiringOutput m_firingOutput;

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
