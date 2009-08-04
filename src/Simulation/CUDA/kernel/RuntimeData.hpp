#ifndef RUNTIME_DATA_HPP
#define RUNTIME_DATA_HPP


#include <stddef.h>
#include <stdint.h>
#include <ctime>
#include "NVector.hpp"
#include "SMatrix.hpp"
#include "kernel.cu_h"

struct RuntimeData
{
	RuntimeData(
			size_t partitionCount,
			size_t maxPartitionSize,
			uint maxDelay,
			size_t maxL0SynapsesPerDelay,
			size_t maxL0RevSynapsesPerNeuron,
			size_t maxL1SynapsesPerDelay,
			size_t maxL1RevSynapsesPerNeuron,
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
	struct FiringProbe* firingProbe;

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

	bool usingSTDP() const;
	void configureSTDP();

	//! \see ::enableSTDP
	void enableSTDP(int tauP, int tauD,
			float* potentiation,
			float* depression,
			float maxWeight);

	float stdpMaxWeight() const;

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

	private :

		uint m_maxDelay;

		uint32_t m_cycle;

		// see kernel.h for enumeration of connectivity matrices
		std::vector<struct ConnectivityMatrix*> m_cm;

		cudaDeviceProp m_deviceProperties;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		/* True if host buffers have not been copied to device */
		bool m_deviceDirty;

		bool m_usingSTDP;

		//! \todo remote tau parameters, can be inferred from vectors
		int m_stdpTauP;
		int m_stdpTauD;
		std::vector<float> m_stdpPotentiation;
		std::vector<float> m_stdpDepression;
		float m_stdpMaxWeight;

		struct timeval m_start; // set before first step
		struct timeval m_end;   // set after every step

		bool m_haveL1Connections;

		// no need for getters for a single use
		friend void configureDevice(RuntimeData*);
};

#endif
