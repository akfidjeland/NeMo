//! \file NVector.hpp

#ifndef N_VECTOR_HPP
#define N_VECTOR_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_array.hpp>

namespace nemo {
	namespace cuda {

/*! \brief Per-neuron data array 
 *
 * Neuron data are organised on a per-partition basis. There are M copies
 * (planes, subvectors) of the data.
 *
 * \author Andreas Fidjeland
 */
template<typename T, int M = 1>
class NVector
{
	public :

		/*! Initialise a 1D parameter vector, potentially for several
		 * partitions. 
		 *
		 * The data is organised in a 2D data struture such that one row
		 * contains all the 1D data for a single partition. This ensures
		 * correct coalescing for 64b data and for 32b data on 1.0 and 1.1
		 * devices. For >=1.2 devices accesses will be coalesced regardless of
		 * whether we use the additional padding a 2D allocation gives.
		 *
		 * \param partitionCount
		 * 		total number of partitionCount simulated on device
		 * \param maxPartitionSize 
		 * 		max size of all partitions in part of network simulated on
		 * 		device 
		 * \param allocHostData
		 * 		by default a host-side copy of the data is created. This can be
		 * 		populated and copied to the device, or conversely be filled by
		 * 		copying *from* the device and then queried.
		 * \param pinHostData
		 * 		if the host side buffer is used for device input or output, it
		 * 		can be allocated as 'pinned' memory. This makes data transfers
		 * 		faster at the cost of reducing the amount of virtual memory
		 * 		available to the host system.
		 */
		NVector(size_t partitionCount,
				size_t maxPartitionSize,
				bool allocHostData,
				bool pinHostData = false);
        
		/*! \return pointer to device data */
		T* deviceData() const;

		/*! \return number of words of data in each subvector, including padding */
		size_t size() const;

		/*! \return number of bytes of data in all vectors, including padding */
		size_t bytes() const;
		size_t d_allocated() const;

		/*! \return word pitch for vector, i.e. number of neurons (including
		 * padding) for each partition */
		size_t wordPitch() const;

		//! \todo remove byte pitch if not in use
		/*! \return byte pitch for vector, i.e. number of neurons (including
		 * padding) for each partition */
		size_t bytePitch() const;

		const T* copyFromDevice();

		/*! Copy entire host buffer to device and deallocote host memory */
		void moveToDevice();
		
		/*! Copy entire host buffer to the device */
		void copyToDevice();
		
		/*! Set row of data (in host buffer) for a single partition */ 
		//! \todo change parameter order, with vector first
		void setPartition(size_t partitionIdx, const T* arr, size_t length, size_t subvector=0);

		/*! Set value (in host buffer) for a single neuron */
		//! \todo change parameter order, with vector first
		void setNeuron(size_t partitionIdx, size_t neuronIdx, const T& val, size_t subvector=0);

		/* Replace host data */
		void set(const std::vector<T>&);

		T getNeuron(size_t partitionIdx, size_t neuronIdx, size_t subvector=0) const;

		/*! Fill all entries in a single plane with the same value */
		void fill(const T& val, size_t subvector=0);

	private :

		boost::shared_array<T> m_deviceData;
		boost::shared_array<T> m_hostData;

		const size_t m_partitionCount;

		size_t m_pitch;

		size_t offset(size_t subvector, size_t partitionIdx, size_t neuronIdx) const;
};

}	} // end namespace

#include "NVector.ipp"


#endif
