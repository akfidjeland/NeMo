//! \file SMatrix.hpp
#ifndef S_MATRIX_HPP
#define S_MATRIX_HPP

#include <vector>

/*! \brief Sparse synapse matrix 
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
		 * \param planeCount
		 * 		Several planes of the same dimensions can be allocated
		 * 		back-to-back with the distance between them returned by size().
		 * 		This reduces the number of parameters that need to be passed to
		 * 		the kernel.
		 * \param allocHostData
		 * 		If true, allocate data on the host-side. If the data structure
		 * 		is only used internally in the kernel, this is wasteful.
		 */
		SMatrix(size_t partitionCount,
				size_t maxPartitionSize,
				size_t maxDelay,
                size_t maxSynapsesPerDelay,
				bool allocHostData,
				size_t planeCount=1);

		~SMatrix();

		/*! \return word pitch for each row of delay data */
		size_t delayPitch() const;

		/*! Add a synapse to the given row. Return the length of the row after
		 * addition */
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
			size_t plane=0);

		/*! Copy entire host buffer to the device */
		void copyToDevice();

		/*! Copy a single plane from device to host. Data are valid until the
		 * next call to copyToHost. */
		const T* copyToHost(size_t plane);

        /*! Copy entire host buffer to device and clear it (host-side) */
        void moveToDevice();

		/*! \return number of words of data in each plane, including padding */
		size_t size() const;

        T* d_data() const;

		/*! Fill plane with value */
		void h_fill(const T& val, size_t plane=0);
		void d_fill(const T& val, size_t plane=0);

		/*! Look up data on the host */
		const T& h_lookup(size_t sourcePartition,
				size_t sourceNeuron,
				size_t delay,
				size_t synapseIndex,
				size_t plane=0) const;

		/*! Look up data on the device. This can be very slow. */
		T d_lookup(size_t sourcePartition,
				size_t sourceNeuron,
				size_t delay,
				size_t synapseIndex,
				size_t plane=0) const;

		/*! \return word offset */
		size_t offset(
				size_t sourcePartition,
				size_t sourceNeuron,
				size_t delay,
				size_t synapseIndex,
				size_t plane=0) const;

	private :

		/*! \return number of bytes of data for all sub-matrices, including padding */
		size_t bytes() const;

		T* m_deviceData;

		/*!  \todo store host data separately for each plane, in moveToDevice,
		 * clear some of these planes, but leave the forward address and
		 * weights which we may need to read back later */
		std::vector<T> m_hostData;

		std::vector<size_t> m_rowLength;

		size_t m_partitionCount;

		size_t m_maxPartitionSize;

		size_t m_maxSynapsesPerDelay;

		size_t m_maxDelay;

		size_t m_pitch;

		size_t m_planeCount;

		/*! \return word offset into length array */
		size_t lenOffset(
				size_t sourcePartition,
				size_t sourceNeuron,
				size_t delay) const;
};


#include "SMatrix.ipp"

#endif
