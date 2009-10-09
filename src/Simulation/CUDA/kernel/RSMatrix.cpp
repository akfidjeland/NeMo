#include "RSMatrix.hpp"
#include "connectivityMatrix.cu_h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdexcept>


RSMatrix::RSMatrix(size_t partitionSize) :
	m_hostData(partitionSize),
	m_partitionSize(partitionSize),
	m_pitch(0),
	m_allocated(0)
{ }



/*! Allocate device memory and a linear block of host memory of the same size */
boost::shared_ptr<uint32_t>&
RSMatrix::allocateDeviceMemory()
{
	size_t desiredPitch = maxSynapsesPerNeuron() * sizeof(uint32_t);
	size_t height = RCM_SUBMATRICES * m_partitionSize;
	size_t bytePitch = 0;

	uint32_t* deviceData = NULL;
	CUDA_SAFE_CALL(
			cudaMallocPitch((void**) &deviceData,
				&bytePitch,
				desiredPitch,
				height));
	m_pitch = bytePitch / sizeof(uint32_t);
	m_allocated = bytePitch * height;

	CUDA_SAFE_CALL(cudaMemset2D((void*) deviceData,
				bytePitch, 0, bytePitch, height));

	m_deviceData = boost::shared_ptr<uint32_t>(deviceData , cudaFree);
	return m_deviceData;
}



bool
RSMatrix::onDevice() const
{
	return m_deviceData.get() != NULL;
}



size_t
RSMatrix::planeSize() const
{
	return m_partitionSize * m_pitch;
}



size_t
RSMatrix::maxSynapsesPerNeuron() const
{
	size_t n = 0;
	for(host_sparse_t::const_iterator i = m_hostData.begin();
			i != m_hostData.end(); ++i) {
		n = std::max(n, i->size());
	}
	return n;
}



//! \todo pass in parameters here
void
RSMatrix::copyToDevice(host_sparse_t h_mem, uint32_t* d_mem)
{
	/* We only need to store the addresses on the host side */
	std::vector<uint32_t> buf(planeSize(), INVALID_REVERSE_SYNAPSE);
	for(host_sparse_t::const_iterator n = h_mem.begin(); n != h_mem.end(); ++n) {
		size_t offset = (n - h_mem.begin()) * m_pitch;
		std::copy(n->begin(), n->end(), buf.begin() + offset);
	}

	CUDA_SAFE_CALL(
			cudaMemcpy(
				d_mem + RCM_ADDRESS * planeSize(),
				&buf[0],
				planeSize() * sizeof(uint32_t),
				cudaMemcpyHostToDevice));

}


void
RSMatrix::moveToDevice()
{
	boost::shared_ptr<uint32_t> d_mem = allocateDeviceMemory();
	copyToDevice(m_hostData, d_mem.get());
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
	/*! \note we cannot check source partition or neuron here, since this class
	 * only deals with the reverse synapses for a single partition. It should
	 * be checked in the caller */
	uint32_t synapse = r_packSynapse(sourcePartition, sourceNeuron, sourceSynapse, delay);
	m_hostData.at(targetNeuron).push_back(synapse);
}



void
RSMatrix::clearStdpAccumulator()
{
	if(!onDevice()) {
		throw std::logic_error("attempting to clear STDP array before device memory allocated");
	}

	CUDA_SAFE_CALL(cudaMemset2D(d_stdp(), m_pitch*sizeof(uint32_t),
				0, m_pitch*sizeof(uint32_t), m_partitionSize));
}



size_t
RSMatrix::d_allocated() const
{
	return m_allocated;
}



uint32_t*
RSMatrix::d_address() const
{
	return m_deviceData.get() + RCM_ADDRESS * planeSize();
}



float*
RSMatrix::d_stdp() const
{
	return (float*) m_deviceData.get() + RCM_STDP * planeSize();
}
