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
#include <SimulationBackend.hpp>

namespace nemo {

	class ConfigurationImpl;
	class NetworkImpl;

	namespace cuda {

class Simulation : public nemo::SimulationBackend
{
	public :

		Simulation(
				const nemo::NetworkImpl& net,
				const nemo::ConfigurationImpl& conf);

		~Simulation();

		/*! Select device (for this thread) if a device with the minimum
		 * required characteristics is present on the host system.
		 *
		 * \return device number or -1 if no suitable device found */
		//! \todo move this to configuration
		static int selectDevice();

		/*! Set the device (for this thread) if the chosen device exists and
		 * meets the minimum required capabilities.
		 *
		 * \return
		 * 		-1 if the chosen device is not suitable
		 * 		\a dev otherwise
		 */
		static int setDevice(int dev);

		/*
		 * CONFIGURATION
		 */

		/*! \copydoc nemo::SimulationBackend::getFiringBufferLength */
		unsigned getFiringBufferLength() const; 

		unsigned getFractionalBits() const;

		/*
		 * NETWORK SIMULATION
		 */

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& nidx);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<fix_t>& current);

		/*! \copydoc nemo::SimulationBackend::step */
		void step();

		/*! \copydoc nemo::SimulationBackend::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::SimulationBackend::getSynapses */
		void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic);

		/*! \copydoc nemo::SimulationBackend::readFiring */
		unsigned readFiring(const std::vector<unsigned>** cycles, const std::vector<unsigned>** nidx);

		/*! \copydoc nemo::SimulationBackend::flushFiringBuffer */
		void flushFiringBuffer();

		void finishSimulation();

		/*
		 * TIMING
		 */

		/*! \copydoc nemo::SimulationBackend::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::SimulationBackend::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::SimulationBackend::resetTimer */
		void resetTimer();

		static unsigned defaultPartitionSize();
		static unsigned defaultFiringBufferLength();

	private :

		class SimulationImpl* m_impl;

};


	} // end namespace cuda
} // end namespace nemo

#endif
