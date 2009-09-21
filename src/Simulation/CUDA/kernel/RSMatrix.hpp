#ifndef RS_MATRIX_HPP
#define RS_MATRIX_HPP

//! \file RSMatrix.hpp

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "kernel.cu_h"

/*! \brief Sparse synapse matrix in reverse format for a single partition
 *
 * Synapses in this matrix are stored on a per-target basis. Unlike in SMatrix
 * there is no further sub-division into separate delays.
 *
 * The reverse matrix has two planes: one for reverse addressing and one for
 * accumulating STDP statistics (LTP and LTD).
 *
 * \see SMatrix
 *
 * \author Andreas Fidjeland
 */
struct RSMatrix
{
	public:

		RSMatrix(size_t partitionSize, size_t maxSynapsesPerNeuron);

		void addSynapse(
				unsigned int sourcePartition,
				unsigned int sourceNeuron,
				unsigned int sourceSynapse,
				unsigned int targetNeuron,
				unsigned int delay);

		void moveToDevice();

		void clearStdpAccumulator();

		/*! \return bytes allocated on the device */
		size_t d_allocated() const;

		/*! \return word pitch */
		size_t pitch() const { return m_pitch; }

		/*! \return device address of reverse address matrix */
		uint32_t* d_address() const;

		/*! \return device address of STDP accumulator matrix */
		float* d_stdp() const;

	private:

		boost::shared_ptr<uint32_t> m_deviceData;

		std::vector<uint32_t> m_hostData;

		size_t m_partitionSize;

		size_t m_maxSynapsesPerNeuron;

		std::vector<size_t> m_synapseCount; // per neuron

		size_t m_pitch;

		size_t m_allocated;

		/* Indices of the two planes of the matrix */
		enum {
			RCM_ADDRESS = 0,
			RCM_STDP,
			RCM_SUBMATRICES
		};

		/*! \return size (in words) of a single plane of the matrix */
		size_t size() const;

};

#endif
