#include <algorithm>
#include <assert.h>

#include "exception.hpp"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {


template<typename T, int M>
NVector<T, M>::NVector(
		size_t partitionCount,
		size_t maxPartitionSize,
		bool allocHostData) :
	m_hostData(NULL),
	m_partitionCount(partitionCount),
	m_pitch(0)
{
	size_t height = M * partitionCount;
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


template<typename T, int M>
NVector<T, M>::~NVector()
{
	d_free(m_deviceData);
	if(m_hostData != NULL) {
		delete[] m_hostData;
	}
}


template<typename T, int M>
T*
NVector<T, M>::deviceData() const
{
	return m_deviceData;
}


template<typename T, int M>
size_t
NVector<T, M>::size() const
{
	return m_partitionCount * m_pitch;
}


template<typename T, int M>
size_t
NVector<T, M>::bytes() const
{
	return M * size() * sizeof(T);
}


template<typename T, int M>
size_t
NVector<T, M>::d_allocated() const
{
	return bytes();
}



template<typename T, int M>
size_t
NVector<T, M>::wordPitch() const
{
	return m_pitch;
}


template<typename T, int M>
size_t
NVector<T, M>::bytePitch() const
{
	return m_pitch * sizeof(T);
}


template<typename T, int M>
const T*
NVector<T, M>::copyFromDevice()
{
	memcpyFromDevice(m_hostData, m_deviceData, M * size());
	return m_hostData;
}


template<typename T, int M>
void
NVector<T, M>::moveToDevice()
{
	copyToDevice();
	delete[] m_hostData;
	m_hostData = NULL;
}


template<typename T, int M>
void
NVector<T, M>::copyToDevice()
{
	memcpyToDevice(m_deviceData, m_hostData, M * size());
}



template<typename T, int M>
size_t
NVector<T, M>::offset(size_t subvector, size_t partitionIdx, size_t neuronIdx) const
{
	//! \todo thow exception if incorrect size is used
	assert(subvector < M);
	assert(partitionIdx < m_partitionCount);
	assert(neuronIdx < m_pitch);
	return (subvector * m_partitionCount + partitionIdx) * m_pitch + neuronIdx;
}


template<typename T, int M>
void
NVector<T, M>::setPartition(size_t partitionIdx, const T* data, size_t length, size_t subvector)
{
	std::copy(data, data + length, m_hostData + offset(subvector, partitionIdx, 0));
}


template<typename T, int M>
void
NVector<T, M>::setNeuron(size_t partitionIdx, size_t neuronIdx, const T& val, size_t subvector)
{
    m_hostData[offset(subvector, partitionIdx, neuronIdx)] = val;
}



template<typename T, int M>
T
NVector<T, M>::getNeuron(size_t partitionIdx, size_t neuronIdx, size_t subvector) const
{
    return m_hostData[offset(subvector, partitionIdx, neuronIdx)];
}



template<typename T, int M>
void
NVector<T, M>::fill(const T& val, size_t subvector)
{
	std::fill(m_hostData + subvector * m_pitch, m_hostData + (subvector+1) * m_pitch, val);
}

}	} // end namespace
