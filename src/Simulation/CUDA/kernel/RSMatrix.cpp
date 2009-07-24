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
	m_maxPartitionPitch(partitionCount, 0)
{
	size_t height = RCM_SUBMATRICES * partitionCount * maxPartitionSize;
	size_t bytePitch = 0;
	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &m_deviceData,
				&bytePitch,
				maxSynapsesPerNeuron * sizeof(uint32_t),
				height));
	m_pitch = bytePitch / sizeof(uint32_t);
	d_fill(RCM_ADDRESS, 0);
	d_fill(RCM_STDP, 0);

	/* We only need to store the addresses on the host side */
	m_hostData.resize(size(), INVALID_REVERSE_SYNAPSE);
}


RSMatrix::~RSMatrix()
{
    CUDA_SAFE_CALL(cudaFree(m_deviceData));
}



size_t
RSMatrix::pitch() const
{
	return m_pitch;
}


bool
RSMatrix::empty() const
{
	return pitch() == 0;
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



const std::vector<uint>&
RSMatrix::maxPartitionPitch() const
{
	return m_maxPartitionPitch;
}



uint32_t*
RSMatrix::d_data() const
{
	return m_deviceData;
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
