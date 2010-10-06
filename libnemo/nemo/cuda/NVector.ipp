#include <algorithm>
#include <cassert>

#include "exception.hpp"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {


template<typename T, int M>
NVector<T, M>::NVector(
		size_t partitionCount,
		size_t maxPartitionSize,
		bool allocHostData,
		bool pinHostData) :
	m_partitionCount(partitionCount),
	m_pitch(0)
{
	size_t height = M * partitionCount;
	size_t bytePitch = 0;
	void* d_ptr = NULL;
	d_mallocPitch(&d_ptr, &bytePitch, maxPartitionSize * sizeof(T), height, "NVector");
	m_deviceData = boost::shared_array<T>(static_cast<T*>(d_ptr), d_free);

	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	d_memset2D(d_ptr, bytePitch, 0x0, height);

	//! \todo may need a default value here
	if(allocHostData) {
		if(pinHostData) {
			void* h_ptr = NULL;
			mallocPinned(&h_ptr, height * m_pitch * sizeof(T));
			m_hostData = boost::shared_array<T>(static_cast<T*>(h_ptr), freePinned);
		} else {
			m_hostData = boost::shared_array<T>(new T[height * m_pitch]);
		}
	}
}



template<typename T, int M>
T*
NVector<T, M>::deviceData() const
{
	return m_deviceData.get();
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
	memcpyFromDevice(m_hostData.get(), m_deviceData.get(), M * size());
	return m_hostData.get();
}


template<typename T, int M>
void
NVector<T, M>::moveToDevice()
{
	copyToDevice();
	m_hostData.reset();
}


template<typename T, int M>
void
NVector<T, M>::copyToDevice()
{
	memcpyToDevice(m_deviceData.get(), m_hostData.get(), M * size());
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
	std::copy(data, data + length, m_hostData.get() + offset(subvector, partitionIdx, 0));
}


template<typename T, int M>
void
NVector<T, M>::setNeuron(size_t partitionIdx, size_t neuronIdx, const T& val, size_t subvector)
{
    m_hostData[offset(subvector, partitionIdx, neuronIdx)] = val;
}



template<typename T, int M>
void
NVector<T, M>::set(const std::vector<T>& vec)
{
	assert(vec.size() == M * size());
	std::copy(vec.begin(), vec.end(), m_hostData.get());
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
	std::fill(m_hostData.get() + subvector * m_pitch, m_hostData.get() + (subvector+1) * m_pitch, val);
}

}	} // end namespace
