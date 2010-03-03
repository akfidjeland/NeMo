#ifndef RUNTIME_DATA_HPP
#define RUNTIME_DATA_HPP


#include <stddef.h>
#include <stdint.h>

#include "NVector.hpp"
#include "DeviceAssertions.hpp"
#include "Timer.hpp"
#include "STDP.hpp"

namespace nemo {

class RuntimeData
{
	public :

		RuntimeData(bool setReverse, unsigned maxReadPeriod);

		// for debugging purposes, fix the partition size used
		RuntimeData(bool setReverse,
				unsigned maxReadPeriod,
				unsigned maxPartitionSize);

		~RuntimeData();

		/*
		 * CONFIGURATION
		 */

		/*! Switch on logging and send output to stdout */
		void logToStdout();

		/*
		 * NETWORK CONSTRUCTION
		 */

		void addNeuron(unsigned int idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void addSynapses(
				uint source,
				const std::vector<uint>& targets,
				const std::vector<uint>& delays,
				const std::vector<float>& weights,
				//! \todo change to bool
				const std::vector<unsigned char> isPlastic);

		/*
		 * NETWORK SIMULATION
		 */

		void startSimulation();

		void stepSimulation(const std::vector<uint>& fstim)
			throw(DeviceAssertionFailure, std::logic_error);

		void applyStdp(float reward);

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
		uint readFiring(const std::vector<uint>** cycles, const std::vector<uint>** nidx);

		void flushFiringBuffer();

		void finishSimulation();

		/*
		 * TIMING
		 */

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step */
		long int elapsedWallclock();

		void resetTimer();

		STDP<float> stdpFn;
		// should be private, but have problems with friend with C linkage

	private :

		/* At any time the network is in one of three states. Some methods can
		 * only be called in a specific state. */
		//! \todo only set reverse matrix at the end, and merge config and construction
		enum State { CONFIGURING, CONSTRUCTING, SIMULATING, ZOMBIE };
		State m_state;

		void ensureState(State);

		uint32_t* setFiringStimulus(const std::vector<uint>& nidx);

		//! \todo add this to logging output
		/*! \return
		 * 		number of bytes allocated on the device
		 *
		 * It seems that cudaMalloc*** does not fail properly when running out of
		 * memory, so this value could be useful for diagnostic purposes */
		size_t d_allocated() const;

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		class NeuronParameters* m_neurons;

		uint32_t m_cycle;

		struct ConnectivityMatrix* m_cm;

		NVector<uint64_t>* m_recentFiring;

		class ThalamicInput* m_thalamicInput;

		/* Densely packed, one bit per neuron */
		NVector<uint32_t>* m_firingStimulus;

		struct FiringOutput* m_firingOutput;

		struct CycleCounters* m_cycleCounters;

		class DeviceAssertions* m_deviceAssertions;

		//! \todo no need to keep this around
		cudaDeviceProp m_deviceProperties;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		nemo::Timer m_timer;

		unsigned int m_maxReadPeriod;

		void configureStdp();
		bool usingStdp() const;

		bool m_logging;
};

} // end namespace nemo

#endif
