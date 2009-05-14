//! \file SMatrix.hpp
#ifndef S_MATRIX_HPP
#define S_MATRIX_HPP

#include <vector>

/*! \brief Sparse synaptic matrix 
 *
 * Matrix containing per-synapse data. This is a template class and can thus be
 * used for different purposes.
 *
 * The data is split per neuron, and is further sub-divided according to delay.
 * Delays are assumed to be at least 1.
 *
 * Writing to the array is a two-stage process. First, data is written to a
 * host buffer. This can be done in multiple operations. Second, data is
 * transferred to the device.
 *
 * \author Andreas Fidjeland
 */
template<typename T>
struct SMatrix
{
	public :

		/*! 
		 * \param submatrixCount
		 * 		Several submatrices of the same dimensions can be allocated
		 * 		back-to-back with the distance between them retrurned by
		 * 		size(). This reduces the number of parameters that need to be
		 * 		passed to the kernel.
		 * \param allocHostData
		 * 		If true, allocate data on the host-side. If the data structure
		 * 		is only used internally in the kernel, this is wasteful.
		 */
		SMatrix(size_t partitionCount,
				size_t maxPartitionSize,
				size_t maxDelay,
                size_t maxSynapsesPerDelay,
				bool allocHostData,
				size_t submatrixCount=1);

		~SMatrix();

		/*! \return word pitch for each row of delay data */
		size_t delayPitch() const;

        /*! Add a synapse to the given row. Return the length of the row after addition */
        size_t addSynapse(
                size_t sourcePartition,
                size_t sourceNeuron,
                size_t delay,
                const T& data); 

		/*! Set a row of data pertaining to a single presynaptic neuron and a
		 * single delay */
        void setDelayRow(
                size_t sourcePartition,
                size_t sourceNeuron,
                size_t delay,
                const std::vector<T>& data,
				size_t submatrix=0);

		/*! Copy entire host buffer to the device */
		void copyToDevice();

		/*! Clear the host buffer */
		void clearHostBuffer();

        /*! Copy entire host buffer to device and clear it (host-side) */
        void moveToDevice();

		/*! \return number of words of data in each submatrix, including padding */
		size_t size() const;

        T* deviceData() const;

		/*! Set default value (in host buffer) for specific submatrix (the
		 * default is 0) */
		void fillHostBuffer(const T& val, size_t submatrix=0);

	private :

		/*! \return number of bytes of data for all sub-matrices, including padding */
		size_t bytes() const;

		T* m_deviceData;
		std::vector<T> m_hostData;

        std::vector<size_t> m_rowLength;

		size_t m_partitionCount;

		size_t m_maxPartitionSize;

        size_t m_maxDelay;

		size_t m_pitch;

		size_t m_submatrixCount;

        /*! \return word offset */
        size_t offset(
                size_t sourcePartition,
                size_t sourceNeuron,
                size_t delay,
                size_t synapseIndex,
				size_t submatrix=0) const;

        /*! \return word offset into length array */
        size_t lenOffset(
                size_t sourcePartition,
                size_t sourceNeuron,
                size_t delay) const;
};


#include "SMatrix.ipp"

#endif
