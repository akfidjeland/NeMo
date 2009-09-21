#include "RSMatrix.hpp"
#include "connectivityMatrix.cu_h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>


RSMatrix::RSMatrix(
		size_t partitionCount,
		size_t maxPartitionSize,
		size_t maxSynapsesPerNeuron) :
	m_deviceData(NULL),
	m_partitionCount(partitionCount),
	m_maxPartitionSize(maxPartitionSize),
	m_maxSynapsesPerNeuron(maxSynapsesPerNeuron),
	m_synapseCount(partitionCount * maxPartitionSize, 0),
	m_maxPartitionPitch(partitionCount, 0),
	m_allocated(0)
{
	size_t height = RCM_SUBMATRICES * partitionCount * maxPartitionSize;
	size_t bytePitch = 0;

	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &m_deviceData,
				&bytePitch,
				maxSynapsesPerNeuron * sizeof(uint32_t),
				height));
	m_pitch = bytePitch / sizeof(uint32_t);
	m_allocated = bytePitch * height;
	d_fill(RCM_ADDRESS, 0);
	d_fill(RCM_STDP, 0);

	/* We only need to store the addresses on the host side */
	m_hostData.resize(size(), INVALID_REVERSE_SYNAPSE);
}


RSMatrix::~RSMatrix()
{
    CUDA_SAFE_CALL(cudaFree(m_deviceData));
}



bool
RSMatrix::empty() const
{
	return m_pitch == 0;
}

size_t
RSMatrix::size() const
{
	return m_partitionCount * m_maxPartitionSize * m_pitch;
}




void
RSMatrix::moveToDevice()
{
	/* We only need to copy the addresses across */
	CUDA_SAFE_CALL(
			cudaMemcpy(
				m_deviceData + RCM_ADDRESS * size(),
				&m_hostData[0],
				size() * sizeof(uint32_t),
				cudaMemcpyHostToDevice));
	m_hostData.clear();
}



void 
RSMatrix::addSynapse(
		unsigned int sourcePartition,
		unsigned int sourceNeuron,
		unsigned int sourceSynapse,
		unsigned int targetPartition,
		unsigned int targetNeuron,
		unsigned int delay)
{
	assert(targetPartition < m_partitionCount);
	assert(targetNeuron < m_maxPartitionSize);

	size_t addr = targetPartition * m_maxPartitionSize + targetNeuron;
	size_t targetSynapse = m_synapseCount[addr];
	m_synapseCount[addr] += 1;

	m_maxPartitionPitch[targetPartition]
		= std::max(m_maxPartitionPitch[targetPartition], (unsigned int) targetSynapse+1);

	assert(targetSynapse < m_maxSynapsesPerNeuron);
	assert(targetSynapse < m_pitch);

	//! \todo refactor
	size_t synapseAddress
		= targetPartition * m_maxPartitionSize * m_pitch
		+ targetNeuron * m_pitch
		+ targetSynapse;

	assert(synapseAddress < m_hostData.size());
	assert(sourcePartition < m_partitionCount);
	assert(sourceNeuron < m_maxPartitionSize);

	m_hostData[synapseAddress] 
		= r_packSynapse(sourcePartition, sourceNeuron, sourceSynapse, delay);
}




void
RSMatrix::d_fill(size_t plane, char val) const
{
	assert(plane < RCM_SUBMATRICES);
	size_t height = m_partitionCount * m_maxPartitionSize;
	CUDA_SAFE_CALL(cudaMemset2D(
				m_deviceData + plane * size(),
				m_pitch*sizeof(uint32_t), val,
				m_pitch*sizeof(uint32_t), height));
}


size_t
RSMatrix::d_allocated() const
{
	return m_allocated;
}



const std::vector<DEVICE_UINT_PTR_T>
RSMatrix::partitionPitch() const
{
	//! \todo add synapse-packing and modify this function
	return std::vector<DEVICE_UINT_PTR_T>(m_partitionCount, m_pitch);
}



//! \todo rename to partitionBase
/*! compute raw addresses (32-bit) based on offset from a base pointer */
const std::vector<DEVICE_UINT_PTR_T>
RSMatrix::partitionAddress() const
{
	//assert(m_deviceData != NULL);

	//! \todo: look up this data at runtime
	const void* MAX_ADDRESS = (void *) 4294967296; // on device
#ifndef __DEVICE_EMULATION__
	assert(m_deviceData <= MAX_ADDRESS);
#endif

	DEVICE_UINT_PTR_T base = (DEVICE_UINT_PTR_T) reinterpret_cast<uint64_t>(m_deviceData);

	//! \todo add synapse-packing and modify this function
	std::vector<DEVICE_UINT_PTR_T> offset(m_partitionCount, (DEVICE_UINT_PTR_T) base);
	for(uint p=0; p < m_partitionCount; ++p) {
		offset[p] += p * m_maxPartitionSize * m_pitch * sizeof(uint32_t);
	}
	return offset;
}


const std::vector<DEVICE_UINT_PTR_T>
RSMatrix::partitionStdp() const
{
	std::vector<DEVICE_UINT_PTR_T> ret = partitionAddress();
	for(std::vector<DEVICE_UINT_PTR_T>::iterator i = ret.begin();
			i != ret.end(); ++i ) {
		*i += size() * sizeof(uint32_t);
	}
	return ret;
}
