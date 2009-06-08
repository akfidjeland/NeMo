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
		size_t maxL0RevSynapsesPerDelay,
		size_t maxL1SynapsesPerDelay,
		size_t maxL1RevSynapsesPerDelay,
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

	struct L1SpikeQueue* spikeQueue;
	struct FiringProbe* firingProbe;

	NVector<uint32_t>* recentFiring;
	NVector<uint32_t>* recentArrivals;

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

	uint stdpCycle() const;

	bool usingSTDP() const;

	//! \see ::enableSTDP
	void enableSTDP(int tauP, int tauD,
			float alphaP, float alphaD, float maxWeight);

    float stdpMaxWeight() const;

    /*! \return number of milliseconds of wall-clock time elapsed since first
     * simulation step */
    long int elapsed();

    void setStart();

    private :

        uint m_maxDelay;

        uint32_t m_cycle;

		// see kernel.h for enumeration of connectivity matrices
		std::vector<struct ConnectivityMatrix*> m_cm;

		/* For STDP we also need to keep track of the most recent spike
		 * delivery, and thus need to know the cycle number. This is just
		 * relative to the last synapse update, and is not a global counter. */
        uint32_t m_stdpCycle;

        cudaDeviceProp m_deviceProperties;

        void setPitch();

        size_t m_pitch32;

		/* True if host buffers have not been copied to device */
		bool m_deviceDirty;

		bool m_usingSTDP;

		int m_stdpTauP;
		int m_stdpTauD;
		float m_stdpAlphaP;
		float m_stdpAlphaD;
		float m_stdpMaxWeight;

        struct timeval m_start; // set before first step
        struct timeval m_end;   // set after every step

		// no need for getters for a single use
		friend void configureDevice(RuntimeData*);
};

#endif
