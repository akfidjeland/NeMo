#include "RSMatrix.hpp"
#include "connectivityMatrix.cu_h"

#include <cuda_runtime.h>
#include <cutil.h>
#include <assert.h>


RSMatrix::RSMatrix(size_t partitionSize, size_t maxSynapsesPerNeuron) :
	m_partitionSize(partitionSize),
	m_maxSynapsesPerNeuron(maxSynapsesPerNeuron),
	m_synapseCount(partitionSize, 0),
	m_allocated(0)
{
	size_t height = RCM_SUBMATRICES * partitionSize;
	size_t bytePitch = 0;

	uint32_t* deviceData = NULL;
	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &deviceData,
				&bytePitch,
				maxSynapsesPerNeuron * sizeof(uint32_t),
				height));
	m_deviceData = boost::shared_ptr<uint32_t>(deviceData , cudaFree);
	m_pitch = bytePitch / sizeof(uint32_t);
	m_allocated = bytePitch * height;

	CUDA_SAFE_CALL(cudaMemset2D((void*) m_deviceData.get(),
				bytePitch, 0, bytePitch, height));

	/* We only need to store the addresses on the host side */
	m_hostData.resize(size(), INVALID_REVERSE_SYNAPSE);
}



size_t
RSMatrix::size() const
{
	return m_partitionSize * m_pitch;
}



void
RSMatrix::moveToDevice()
{
	/* We only need to copy the addresses across. The STDP accumulator has
	 * already been cleared. */
	CUDA_SAFE_CALL(
			cudaMemcpy(
				m_deviceData.get() + RCM_ADDRESS * size(),
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
		unsigned int targetNeuron,
		unsigned int delay)
{
	//! \todo use exceptions here instead
	assert(targetNeuron < m_partitionSize);

	size_t targetSynapse = m_synapseCount[targetNeuron];
	m_synapseCount[targetNeuron] += 1;

	assert(targetSynapse < m_maxSynapsesPerNeuron);
	assert(targetSynapse < m_pitch);

	size_t synapseAddress = targetNeuron * m_pitch + targetSynapse;
	assert(synapseAddress < m_hostData.size());

	/*! \note we cannot check source partition or neuron here, since this class
	 * only deals with the reverse synapses for a single partition. It should
	 * be * checked in the caller */

	m_hostData[synapseAddress] 
		= r_packSynapse(sourcePartition, sourceNeuron, sourceSynapse, delay);
}



void
RSMatrix::clearStdpAccumulator()
{
	CUDA_SAFE_CALL(cudaMemset2D(
				d_stdp(), m_pitch*sizeof(uint32_t), 0,
				m_pitch*sizeof(uint32_t), m_partitionSize));
}



size_t
RSMatrix::d_allocated() const
{
	return m_allocated;
}



uint32_t*
RSMatrix::d_address() const
{
	return m_deviceData.get() + RCM_ADDRESS * size();
}



float*
RSMatrix::d_stdp() const
{
	return (float*) m_deviceData.get() + RCM_STDP * size();
}
