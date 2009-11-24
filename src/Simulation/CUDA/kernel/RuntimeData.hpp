#ifndef RUNTIME_DATA_HPP
#define RUNTIME_DATA_HPP


#include <stddef.h>
#include <stdint.h>
#include "NVector.hpp"
#include "SMatrix.hpp"
#include "kernel.cu_h"

#include <Timer.hpp>
#include <StdpFunction.hpp>


struct RuntimeData
{
	RuntimeData(
			size_t partitionCount,
			size_t maxPartitionSize,
			uint maxDelay,
			size_t maxL0SynapsesPerDelay,
			size_t maxL1SynapsesPerDelay,
			bool setReverse,
			//! \todo determine the entry size inside allocator
			size_t l1SQEntrySize,
			unsigned int maxReadPeriod);

	~RuntimeData();

	uint maxDelay() const;

	/* Copy data to device if this has not already been done, and clear the
	 * host buffers */
	void moveToDevice();

	/*! \return true if network data/configuration has *not* been copied to the
	 * device */
	bool deviceDirty() const;

	/*! \return true if there are *any* L1 connections, i.e. connections
	 * crossing partition boundaries */
	bool haveL1Connections() const;

	struct L1SpikeQueue* spikeQueue;
	struct FiringOutput* firingOutput;

	NVector<uint64_t>* recentFiring;

	NVector<float>* neuronParameters;

	/* Densely packed, one bit per neuron */
	NVector<uint32_t>* firingStimulus;

	class ThalamicInput* thalamicInput;

	/*! \return connectivity matrix with given index */
	struct ConnectivityMatrix* cm(size_t idx) const;

	size_t maxPartitionSize;
	size_t partitionCount;

	/*! \return word pitch for 32-bit neuron arrays */
	size_t pitch32() const;
	size_t pitch64() const;

	float* setCurrentStimulus(size_t n,
			const int* cidx,
			const int* nidx,
			const float* current);

	uint32_t*
		setFiringStimulus(
				size_t count,
				const int* pidx,
				const int* nidx);

	struct CycleCounters* cycleCounters;

	/*! Update internal cycle counters. This should be called every simulation
	 * cycle */
	void step();

	uint32_t cycle() const;

	bool usingStdp() const;

	/*! \return number of milliseconds of wall-clock time elapsed since first
	 * simulation step */
	long int elapsed();

	void setStart();

	/*! \return
	 * 		number of bytes allocated on the device
	 *
	 * It seems that cudaMalloc*** does not fail properly when running out of
	 * memory, so this value could be useful for diagnostic purposes */
	size_t d_allocated() const;

		class nemo::StdpFunction* stdpFn;
		// should be private, but have problems with friend with C linkage

	private :


		uint m_maxDelay;

		uint32_t m_cycle;

		// see kernel.h for enumeration of connectivity matrices
		std::vector<struct ConnectivityMatrix*> m_cm;

		cudaDeviceProp m_deviceProperties;

		void setPitch();

		size_t m_pitch1;
		size_t m_pitch32;
		size_t m_pitch64;

		/* True if host buffers have not been copied to device */
		bool m_deviceDirty;

		nemo::Timer m_timer;

		bool m_haveL1Connections;

		// no need for getters for a single use
		friend void configureDevice(RuntimeData*);
};

#endif
