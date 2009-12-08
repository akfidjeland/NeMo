#include "util.h"

#include <cuda_runtime.h>
#include <algorithm>
//#include <stdexcept>
#include <assert.h>

template<typename T>
SMatrix<T>::SMatrix(
			size_t partitionCount,
			size_t maxPartitionSize,
			size_t maxDelay,
            size_t maxSynapsesPerDelay,
			bool allocHostData,
			size_t planeCount) :
	m_deviceData(NULL),
	m_partitionCount(partitionCount),
	m_maxPartitionSize(maxPartitionSize),
	m_maxSynapsesPerDelay(maxSynapsesPerDelay),
	m_maxDelay(maxDelay),
	m_pitch(0),
	m_planeCount(planeCount)
{
	size_t width = maxSynapsesPerDelay;
	size_t height = planeCount * partitionCount * maxPartitionSize * maxDelay;
	fprintf(stderr, "SMatrix: height = %u (%u planes of %u partitions each with size %u each with %u delays\n",
			height, planeCount, partitionCount, maxPartitionSize, maxDelay);
	size_t bytePitch = 0;
	fprintf(stderr, "SMatrix: allocating %u bytes (w%u x h%u x %u)\n",
			height * width * sizeof(T),
			height,  width,  sizeof(T));
	CUDA_SAFE_CALL(
			cudaMallocPitch((void**)&m_deviceData, 
				&bytePitch, 
				width * sizeof(T),
				height)); 
	m_pitch = bytePitch / sizeof(T);

	/* Set all space including padding to fixed value. This is important as
	 * some warps may read beyond the end of these arrays. */
	/*! \todo this may fail if we have allocated an inordinate amount of space
	 * for synapses. Report this error. */
	CUDA_SAFE_CALL(cudaMemset2D(m_deviceData, bytePitch, 0x0, bytePitch, height));

	//! \todo may need a default value here
	if(allocHostData) {
		m_hostData.resize(height * m_pitch);
        m_rowLength.resize(height, 0);
	}
}



template<typename T>
SMatrix<T>::~SMatrix()
{
    CUDA_SAFE_CALL(cudaFree(m_deviceData));
}



template<typename T>
size_t
SMatrix<T>::size() const
{
	return m_partitionCount * m_maxPartitionSize * m_maxDelay * m_pitch;
}



template<typename T>
T*
SMatrix<T>::d_data() const
{
    return m_deviceData;
}



template<typename T>
size_t
SMatrix<T>::bytes() const
{
	return m_planeCount * size() * sizeof(T);
}



template<typename T>
size_t
SMatrix<T>::d_allocated() const
{
	return bytes();
}


template<typename T>
size_t
SMatrix<T>::delayPitch() const
{
    return m_pitch;
}



template<typename T>
void
SMatrix<T>::copyToDevice()
{
    assert(m_planeCount * size() <= m_hostData.size());
    //! \todo add back exceptions (requires CUDA 2.2)
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
void
SMatrix<T>::copyToHost(size_t plane, std::vector<T>& hostData)
{
	//if(m_hostData.empty()) {
	//    throw std::logic_error("Attempt to copy from device to empty host vector");
	//}
	//! \todo change back to exceptions instead
	assert(plane < m_planeCount);
	assert(hostData.size() == size());
	/* nvcc chokes with "closing brace of template definition not found" if
	 * CUDA_SAFE_CALL is used in cuda 2.1 */
	/* CUDA_SAFE_CALL( */ cudaMemcpy(
			&hostData[0],
			m_deviceData + plane * size(),
			size() * sizeof(T),
			cudaMemcpyDeviceToHost) /*)*/ ;
}




template<typename T>
void
SMatrix<T>::moveToDevice()
{
	copyToDevice();
	m_hostData.clear();
}



template<typename T>
size_t
SMatrix<T>::offset(
        size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay,
        size_t synapseIndex,
		size_t plane) const
{
    assert(sourcePartition < m_partitionCount);
    assert(sourceNeuron < m_maxPartitionSize);
    assert(delay <= m_maxDelay);
    assert(delay >= 1);
    assert(synapseIndex < delayPitch());
	assert(plane < m_planeCount);
    //! \todo refactor
    //! \todo have this call a method which we share with the kernel as well
    return plane * size()
			+ sourcePartition * m_maxPartitionSize * m_maxDelay * delayPitch()
            + sourceNeuron * m_maxDelay * delayPitch()
            + (delay-1) * delayPitch()
            + synapseIndex;
}



template<typename T>
const T&
SMatrix<T>::h_lookup(size_t srcp,
        size_t srcn, size_t delay, size_t sidx, size_t plane) const
{
    return m_hostData[offset(srcp, srcn, delay, sidx, plane)];
}



template<typename T>
T
SMatrix<T>::d_lookup(size_t srcp,
        size_t srcn, size_t delay, size_t sidx, size_t plane) const
{
	T out;
	CUDA_SAFE_CALL(cudaMemcpy(&out,
			m_deviceData + offset(srcp, srcn, delay, sidx, plane),
			sizeof(out),
			cudaMemcpyDeviceToHost));
	return out;
}


template<typename T>
size_t
SMatrix<T>::lenOffset(
        size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay) const
{
    assert(sourcePartition < m_partitionCount);
    assert(sourceNeuron < m_maxPartitionSize);
    assert(delay <= m_maxDelay);
    assert(delay >= 1);
    //! \todo refactor
    size_t r = sourcePartition * m_maxPartitionSize * m_maxDelay 
        + sourceNeuron * m_maxDelay
        + delay - 1;
    assert(r < m_rowLength.size());
    return r;
}



template<typename T>
void
SMatrix<T>::setDelayRow(
		size_t sourcePartition,
		size_t sourceNeuron,
		size_t delay,
        const std::vector<T>& data,
		size_t plane)
{
	std::copy(data.begin(), data.end(), m_hostData.begin() 
            + offset(sourcePartition, sourceNeuron, delay, 0, plane));
    m_rowLength[lenOffset(sourcePartition, sourceNeuron, delay)] = data.size();
}



template<typename T>
size_t
SMatrix<T>::addSynapse(
        size_t sourcePartition,
        size_t sourceNeuron,
        size_t delay,
        const T& data)
{
    size_t i = lenOffset(sourcePartition, sourceNeuron, delay);
    size_t synapseIndex = m_rowLength[i];
    assert(synapseIndex < delayPitch());
    m_hostData[offset(sourcePartition, sourceNeuron, delay, synapseIndex)] = data;
    m_rowLength[i] += 1;
    return synapseIndex + 1;
}


template<typename T>
void
SMatrix<T>::h_fill(const T& value, size_t plane)
{
	typename std::vector<T>::iterator b = m_hostData.begin() + plane * size();
	std::fill(b, b + size(), value);
}


template<typename T>
void
SMatrix<T>::d_fill(const T& value, size_t plane)
{
    CUDA_SAFE_CALL(
        cudaMemset2D(
            m_deviceData + plane * size(),
            m_pitch,
            value,
            m_maxSynapsesPerDelay,
            m_partitionCount * m_maxPartitionSize * m_maxDelay)
    );
}
