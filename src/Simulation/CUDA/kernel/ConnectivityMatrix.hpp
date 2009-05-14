#ifndef CONNECTIVITY_MATRIX_HPP
#define CONNECTIVITY_MATRIX_HPP

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "SMatrix.hpp"
#include "NVector.hpp"

struct ConnectivityMatrix
{
	public:

		ConnectivityMatrix(
				size_t partitionCount,
				size_t maxPartitionSize,
				size_t maxDelay,
				size_t maxSynapsesPerDelay,
				size_t maxRevSynapsesPerDelay);

		/* Set row in delay-partitioned matrix */
		void setDRow(
				unsigned int sourcePartition,
				unsigned int sourceNeuron,
				unsigned int delay,
				const float* weights,
				const unsigned int* targetPartition,
				const unsigned int* targetNeuron,
				size_t length);

		/* Copy data to device and clear host buffers */
		void moveToDevice();

		// new format: delay specific */
		uint* deviceSynapsesD() const;

		/*! \return row pitch in words for delay-specific connetivity */
		size_t synapsePitchD() const;

		/*! \return number of words (including padding) for each submatrix */
		size_t submatrixSize() const;

		uint32_t* deviceDelayBits() const;

		const std::vector<int>& maxSynapsesPerDelay() const;

		/* REVERSE CONNECTIVITY */

		/*! \return reverse connectivity matrix */
		uint* reverseConnectivity() const;

		/*! \return row pitch in words for reverse matrix */
		size_t reversePitch() const;

		/*! \return number of words (includeing padding) for each reverse submatrix */
		size_t reverseSubmatrixSize() const;

		uint32_t* arrivalBits() const;

		const std::vector<int>& maxReverseSynapsesPerDelay() const;

	private:

		/* submatrix 0: address data + timestamp
		 * submatrix 1: weights
		 */
		SMatrix<uint> m_synapses;

		/* We also accumulate the firing delay bits that are used in the spike
		 * delivery */
		NVector<uint32_t> m_delayBits;

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		unsigned int m_maxDelay;

		/* As we fill the matrix, we accumulate per-partition statistics which
		 * can be used for later configuration */
		std::vector<int> m_maxSynapsesPerDelay;
		std::vector<int> m_maxReverseSynapsesPerDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and (dynamic) spike arrival time */
		SMatrix<uint> m_reverse;

		/* Furthermore, to reduce the number of reverse lookups we keep track
		 * of the possible delays at which spikes arrive at each neuron */
		NVector<uint32_t> m_arrivalBits;
};

#endif
