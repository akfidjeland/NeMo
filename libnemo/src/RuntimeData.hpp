#ifndef RUNTIME_DATA_HPP
#define RUNTIME_DATA_HPP


#include <stddef.h>
#include <stdint.h>
#include "NVector.hpp"
#include "DeviceAssertions.hpp"

#include <Timer.hpp>
#include <STDP.hpp>


class RuntimeData
{
	public :

		RuntimeData(
				size_t maxPartitionSize,
				bool setReverse,
				unsigned int maxReadPeriod);

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

		void stepSimulation(size_t fstimCount, const uint* fstimIdx)
			throw(DeviceAssertionFailure, std::logic_error);

		void applyStdp(float reward);

	//! \todo fix accesses in libnemo.cpp and make this private
		//! \todo expose this using std::vector. Deal with raw pointers in the c layer
		void readFiring(uint** cycles, uint** neuronIdx, uint* nfired, uint* ncycles);

		void flushFiringBuffer();

		/*
		 * TIMING
		 */

		/*! \return number of milliseconds of wall-clock time elapsed since first
		 * simulation step */
		long int elapsed();

		void setStart();

		void printCycleCounters();

		class nemo::STDP<float> stdpFn;
		// should be private, but have problems with friend with C linkage

		/* Force all asynchronously launced kernels to complete before returning */
		void syncSimulation();

	private :

		uint32_t* setFiringStimulus(size_t count, const uint* nidx);

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

#endif
