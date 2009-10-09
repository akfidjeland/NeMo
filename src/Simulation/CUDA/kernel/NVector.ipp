#include "util.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <assert.h>
//! \todo add exceptions
//#include <stdexcept>


template<typename T>
NVector<T>::NVector(
		size_t partitionCount,
		size_t maxPartitionSize,
		bool allocHostData,
		size_t subvectorCount) :
	m_partitionCount(partitionCount),
	m_pitch(0),
	m_subvectorCount(subvectorCount)
{
	size_t height = subvectorCount * partitionCount;
	size_t bytePitch = 0;
	CUDA_SAFE_CALL(
			cudaMallocPitch(
				(void**)&m_deviceData,
				&bytePitch,
				maxPartitionSize * sizeof(T),
				height)
	); 
	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	CUDA_SAFE_CALL(cudaMemset2D(m_deviceData, bytePitch, 0x0, bytePitch, height));

	//! \todo may need a default value here
	if(allocHostData) {
		m_hostData.resize(height * m_pitch);
	}
}


template<typename T>
NVector<T>::~NVector()
{
    CUDA_SAFE_CALL(cudaFree(m_deviceData));
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
const std::vector<T>& 
NVector<T>::copyFromDevice()
{
#if 0
	//! \todo allocate on-demand here
	if(m_deviceData->size() > size()) {
		throw std::logic_error("Attempt to read device data into too small host buffer");
	}
#endif
	CUDA_SAFE_CALL(cudaMemcpy(
				&m_hostData[0],
				m_deviceData,
				bytes(),
				cudaMemcpyDeviceToHost));
	return m_hostData;
}


template<typename T>
void
NVector<T>::moveToDevice()
{
	copyToDevice();
	m_hostData.clear();
}


template<typename T>
void
NVector<T>::copyToDevice()
{
#if 0
	if(size() > m_hostData.size()) {
		throw std::logic_error("Attempt to copy Insuffient host data to device");
	}
#endif

	CUDA_SAFE_CALL(
			cudaMemcpy(
				m_deviceData,
				&m_hostData[0],
				bytes(),
				cudaMemcpyHostToDevice));
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
	std::copy(data, data + length, m_hostData.begin() + offset(subvector, partitionIdx, 0));
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
