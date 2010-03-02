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
		RuntimeData(bool setReverse, unsigned maxReadPeriod, unsigned maxPartitionSize);

		~RuntimeData();

		/*
		 * NETWORK CONSTRUCTION
		 */

		void addNeuron(unsigned int idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		void addSynapses(
				//! \todo use std::vector here?
				uint source,
				uint targets[],
				uint delays[],
				float weights[],
				//! \todo use bool here. Transform input in the C layer
				unsigned char is_plastic[],
				size_t length);

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

		/*
		 * TIMING
		 */

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step */
		long int elapsed();

		void setStart();

		void printCycleCounters();

		STDP<float> stdpFn;
		// should be private, but have problems with friend with C linkage

		/* Force all asynchronously launced kernels to complete before returning */
		void syncSimulation();

	private :

		uint32_t* setFiringStimulus(const std::vector<uint>& nidx);

		/*! \return
		 * 		number of bytes allocated on the device
		 *
		 * It seems that cudaMalloc*** does not fail properly when running out of
		 * memory, so this value could be useful for diagnostic purposes */
		size_t d_allocated() const;

		/*! \return true if network data/configuration has *not* been copied to the
		 * device */
		bool deviceDirty() const;

		/* Copy data to device if this has not already been done, and clear the
		 * host buffers */
		void moveToDevice();

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

		cudaDeviceProp m_deviceProperties;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		/* True if host buffers have not been copied to device */
		bool m_deviceDirty;

		nemo::Timer m_timer;

		unsigned int m_maxReadPeriod;

		void configureStdp();
		bool usingStdp() const;
};

} // end namespace nemo

#endif
