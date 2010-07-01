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

		unsigned getFiringBufferLength() const; 

		/*
		 * NETWORK SIMULATION
		 */

		void setFiringStimulus(const std::vector<unsigned>& nidx);
		void setCurrentStimulus(const std::vector<float>& current);
		void step();

		void applyStdp(float reward);

		void getSynapses(unsigned sourceNeuron,
				const std::vector<unsigned>** targetNeuron,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic);

		/*! Read all firing data buffered on the device since the previous
		 * call to this function (or the start of simulation if this is the
		 * first call). The return vectors are valid until the next call to
		 * this function.
		 *
		 * \return
		 * 		Total number of cycles for which we return firing. The caller
		 * 		would most likely already know what this should be, so can use
		 * 		this for sanity checking.
		 */
		unsigned readFiring(const std::vector<unsigned>** cycles, const std::vector<unsigned>** nidx);

		void flushFiringBuffer();

		void finishSimulation();

		/*
		 * TIMING
		 */

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step (or last timer reset) */
		unsigned long elapsedWallclock() const;

		/*! \return number of milliseconds of simulated time elapsed since first
		 * simulation step (or last timer reset) */
		unsigned long elapsedSimulation() const;

		/*! Reset both wall-clock and simulation timer */
		void resetTimer();

		static unsigned defaultPartitionSize();
		static unsigned defaultFiringBufferLength();

	private :

		class SimulationImpl* m_impl;

};


	} // end namespace cuda
} // end namespace nemo

#endif
