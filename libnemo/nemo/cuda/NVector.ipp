#include <algorithm>
#include <assert.h>

#include "exception.hpp"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {


template<typename T>
NVector<T>::NVector(
		size_t partitionCount,
		size_t maxPartitionSize,
		bool allocHostData,
		size_t subvectorCount) :
	m_hostData(NULL),
	m_partitionCount(partitionCount),
	m_pitch(0),
	m_subvectorCount(subvectorCount)
{
	size_t height = subvectorCount * partitionCount;
	size_t bytePitch = 0;
	d_mallocPitch((void**)&m_deviceData, &bytePitch,
				maxPartitionSize * sizeof(T), height, "NVector");
	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */

	d_memset2D(m_deviceData, bytePitch, 0x0, height);

	//! \todo may need a default value here
	if(allocHostData) {
		m_hostData = new T[height * m_pitch];
	}
}


template<typename T>
NVector<T>::~NVector()
{
	d_free(m_deviceData);
	if(m_hostData != NULL) {
		delete[] m_hostData;
	}
}


template<typename T>
T*
NVector<T>::deviceData() const
{
	return m_deviceData;
}


template<typename T>
size_t
NVector<T>::size() const
{
	return m_partitionCount * m_pitch;
}


template<typename T>
size_t
NVector<T>::bytes() const
{
	return m_subvectorCount * size() * sizeof(T);
}


template<typename T>
size_t
NVector<T>::d_allocated() const
{
	return bytes();
}



template<typename T>
size_t
NVector<T>::wordPitch() const
{
	return m_pitch;
}


template<typename T>
size_t
NVector<T>::bytePitch() const
{
	return m_pitch * sizeof(T);
}


template<typename T>
const T*
NVector<T>::copyFromDevice()
{
	memcpyFromDevice(m_hostData, m_deviceData, m_subvectorCount * size());
	return m_hostData;
}


template<typename T>
void
NVector<T>::moveToDevice()
{
	copyToDevice();
	delete[] m_hostData;
	m_hostData = NULL;
}


template<typename T>
void
NVector<T>::copyToDevice()
{
	memcpyToDevice(m_deviceData, m_hostData, m_subvectorCount * size());
}



template<typename T>
size_t
NVector<T>::offset(size_t subvector, size_t partitionIdx, size_t neuronIdx) const
{
	//! \todo thow exception if incorrect size is used
	assert(subvector < m_subvectorCount);
	assert(partitionIdx < m_partitionCount);
	assert(neuronIdx < m_pitch);
	return (subvector * m_partitionCount + partitionIdx) * m_pitch + neuronIdx;
}


template<typename T>
void
NVector<T>::setPartition(size_t partitionIdx, const T* data, size_t length, size_t subvector)
{
	std::copy(data, data + length, m_hostData + offset(subvector, partitionIdx, 0));
}


template<typename T>
void
NVector<T>::setNeuron(size_t partitionIdx, size_t neuronIdx, const T& val, size_t subvector)
{
    m_hostData[offset(subvector, partitionIdx, neuronIdx)] = val;
}



template<typename T>
T
NVector<T>::getNeuron(size_t partitionIdx, size_t neuronIdx, size_t subvector) const
{
    return m_hostData[offset(subvector, partitionIdx, neuronIdx)];
}



template<typename T>
void
NVector<T>::fill(const T& val, size_t subvector)
{
	std::fill(m_hostData + subvector * m_pitch, m_hostData + (subvector+1) * m_pitch, val);
}

}	} // end namespace
