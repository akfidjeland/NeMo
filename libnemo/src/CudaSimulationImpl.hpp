#ifndef NEMO_CUDA_SIMULATION_IMPL_HPP
#define NEMO_CUDA_SIMULATION_IMPL_HPP

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
#include <STDP.hpp>
#include <Timer.hpp>
#include <types.h>
#include <ConfigurationImpl.hpp>

#include "DeviceIdx.hpp"
#include "NVector.hpp"
#include "ConnectivityMatrix.hpp"
#include "DeviceAssertions.hpp"
#include "FiringOutput.hpp"
#include "NeuronParameters.hpp"

namespace nemo {

	class NetworkImpl;

	namespace cuda {

class SimulationImpl
{
	public :

		SimulationImpl(
				const nemo::NetworkImpl& net,
				const nemo::ConfigurationImpl& conf);

		~SimulationImpl();

		/* CONFIGURATION */

		static int selectDevice();
		static int setDevice(int dev);

		static unsigned defaultPartitionSize();
		static unsigned defaultFiringBufferLength();

		unsigned getFiringBufferLength() const { return m_conf.cudaFiringBufferLength(); }

		/* NETWORK SIMULATION */

		void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());

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

		Mapper m_mapper;

		nemo::ConfigurationImpl m_conf;

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

		ConnectivityMatrix m_cm;

		NVector<uint64_t>* m_recentFiring;

		class ThalamicInput* m_thalamicInput;

		/* Densely packed, one bit per neuron */
		NVector<uint32_t>* m_firingStimulus;

		/* The firing buffer keeps data for a certain duration. One bit is
		 * required per neuron (regardless of whether or not it's firing */
		FiringOutput m_firingOutput;

		class CycleCounters* m_cycleCounters;

		class DeviceAssertions* m_deviceAssertions;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		STDP<float> m_stdpFn;
		void configureStdp(const STDP<float>& stdp);
		bool usingStdp() const;

		static int s_device;

		Timer m_timer;
};

	} // end namespace cuda
} // end namespace nemo

#endif
