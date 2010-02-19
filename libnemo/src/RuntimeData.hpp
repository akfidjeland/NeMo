#ifndef RUNTIME_DATA_HPP
#define RUNTIME_DATA_HPP


#include <stddef.h>
#include <stdint.h>
#include "NVector.hpp"
#include "kernel.cu_h"

#include <Timer.hpp>
#include <STDP.hpp>


struct RuntimeData
{
	RuntimeData(
			size_t maxPartitionSize,
			bool setReverse,
			unsigned int maxReadPeriod);

	~RuntimeData();

	uint maxDelay() const;

	/* Copy data to device if this has not already been done, and clear the
	 * host buffers */
	void moveToDevice();

	/*! \return true if network data/configuration has *not* been copied to the
	 * device */
	bool deviceDirty() const;

	struct FiringOutput* firingOutput;

	NVector<uint64_t>* recentFiring;

	float* d_neurons() const;

	size_t neuronVectorLength() const;

	void addNeuron(unsigned int idx,
			float a, float b, float c, float d,
			float u, float v, float sigma);

	/* Densely packed, one bit per neuron */
	NVector<uint32_t>* firingStimulus;

	class ThalamicInput* thalamicInput;

	struct ConnectivityMatrix* cm() const;

	size_t partitionCount() const { return m_partitionCount; }

	size_t maxPartitionSize;

	/*! \return word pitch for 32-bit neuron arrays */
	size_t pitch32() const;
	size_t pitch64() const;

	float* setCurrentStimulus(size_t n,
			const int* cidx,
			const int* nidx,
			const float* current);

	uint32_t* setFiringStimulus(size_t count, const int* nidx);

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

		class nemo::STDP<float> stdpFn;
		// should be private, but have problems with friend with C linkage

		/* Force all asynchronously launced kernels to complete before returning */
		void syncSimulation();

	private :

		class NeuronParameters* m_neurons;

		uint32_t m_cycle;

		struct ConnectivityMatrix* m_cm;

		cudaDeviceProp m_deviceProperties;

		void setPitch();

		size_t m_pitch32;
		size_t m_pitch64;

		size_t m_partitionCount;

		/* True if host buffers have not been copied to device */
		bool m_deviceDirty;

		nemo::Timer m_timer;

		unsigned int m_maxReadPeriod;

		void configureStdp();

		// no need for getters for a single use
		friend void configureDevice(RuntimeData*);
};

#endif
