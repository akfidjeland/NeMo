#ifndef CUDA_NETWORK_HPP
#define CUDA_NETWORK_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stddef.h>

#include <nemo_config.h>
#include "NVector.hpp"
#include "Configuration.hpp"
#include "ConnectivityMatrix.hpp"
#include "DeviceAssertions.hpp"
#include "NeuronParameters.hpp"
#ifdef INCLUDE_TIMING_API
#include "Timer.hpp"
#endif
#include "Simulation.hpp"
#include "STDP.hpp"
#include "types.h"

namespace nemo {

	class Network;

	namespace cuda {

class CudaNetwork : public Simulation
{
	public :

		CudaNetwork(const nemo::Network& net, const nemo::Configuration& conf=Configuration());

		~CudaNetwork();

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

		unsigned getFiringBufferLength() const { return m_conf.cudaFiringBufferLength(); }

		/*
		 * NETWORK SIMULATION
		 */

		void stepSimulation(const std::vector<unsigned>& fstim = std::vector<unsigned>());

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

#ifdef INCLUDE_TIMING_API
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
#endif

		static unsigned defaultPartitionSize();
		static unsigned defaultFiringBufferLength();

	private :

		nemo::Configuration m_conf;

		uint32_t* setFiringStimulus(const std::vector<unsigned>& nidx);

		//! \todo add this to logging output
		/*! \return
		 * 		number of bytes allocated on the device
		 *
		 * It seems that cudaMalloc*** does not fail properly when running out of
		 * memory, so this value could be useful for diagnostic purposes */
		size_t d_allocated() const;

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		NeuronParameters m_neurons;

		uint32_t m_cycle;

		ConnectivityMatrix m_cm;

		NVector<uint64_t>* m_recentFiring;

		class ThalamicInput* m_thalamicInput;

		/* Densely packed, one bit per neuron */
		NVector<uint32_t>* m_firingStimulus;

		/* The firing buffer keeps data for a certain duration. One bit is
		 * required per neuron (regardless of whether or not it's firing */
		class FiringOutput* m_firingOutput;

		class CycleCounters* m_cycleCounters;

		class DeviceAssertions* m_deviceAssertions;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

#ifdef INCLUDE_TIMING_API
		nemo::Timer m_timer;
#endif

		STDP<float> m_stdpFn;
		void configureStdp(const STDP<float>& stdp);
		bool usingStdp() const;

		static int s_device;
};

	} // end namespace cuda
} // end namespace nemo

#endif
