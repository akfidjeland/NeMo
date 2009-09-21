//! \file RSMatrix.hpp

#ifndef RS_MATRIX_HPP
#define RS_MATRIX_HPP

#include <stdint.h>
#include <stddef.h>
#include <vector>

#include "kernel.cu_h"

/*! \brief Sparse synapse matrix in reverse format
 *
 * Synapses in this matrix are stored on a per-target basis. Unlike in SMatrix
 * there is no further sub-division into separate delays.
 *
 * \see SMatrix
 *
 * \author Andreas Fidjeland
 */
struct RSMatrix
{
	public:

		RSMatrix(size_t partitionCount,
				size_t maxPartitionSize,
				size_t maxSynapsesPerNeuron);

		~RSMatrix();

		void addSynapse(
				unsigned int sourcePartition,
				unsigned int sourceNeuron,
				unsigned int sourceSynapse,
				unsigned int targetPartition,
				unsigned int targetNeuron,
				unsigned int delay);

		void moveToDevice();

		void d_fill(size_t plane, char val) const;

		/*! \return bytes allocated on the device */
		size_t d_allocated() const;

		bool empty() const;

		/*! \return pitch for per-partition addressing */
		const std::vector<DEVICE_UINT_PTR_T> partitionPitch() const;

		/*! \return addresses to each partition's reverse address data */
		const std::vector<DEVICE_UINT_PTR_T> partitionAddress() const;

		/*! \return addresses to each partition's STDP accumulator data */
		const std::vector<DEVICE_UINT_PTR_T> partitionStdp() const;

	private:

		//! \todo remove altogher
		size_t size() const; /*! \return size (in words) of a single plane of the matrix */

		uint32_t* m_deviceData;

		std::vector<uint32_t> m_hostData;

		size_t m_partitionCount;

		size_t m_maxPartitionSize;

		size_t m_maxSynapsesPerNeuron;

		std::vector<size_t> m_synapseCount; // per neuron

		std::vector<uint> m_maxPartitionPitch;

		size_t m_pitch;

		size_t m_allocated;
};

#endif
