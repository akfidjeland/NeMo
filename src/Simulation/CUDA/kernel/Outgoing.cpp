#include "Outgoing.hpp"

#include <vector>
#include "boost/tuple/tuple_comparison.hpp"
#include <cuda_runtime.h>

#include "kernel.cu_h"
#include "util.h"


void 
Outgoing::addSynapseGroup(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		pidx_t targetPartition)
{
	key_t key(sourcePartition, sourceNeuron);
	m_acc[key].insert(make_outgoing(targetPartition, delay));
}



size_t
Outgoing::maxPitch() const
{
	size_t pitch = 0;
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		pitch = std::max(pitch, i->second.size());
	}
	return pitch;
}



void
Outgoing::moveToDevice(size_t partitionCount)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE;
	size_t width = maxPitch() * sizeof(outgoing_t);

	// allocate device memory for table
	outgoing_t* d_arr;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_arr, &m_pitch, width, height));
	md_arr = boost::shared_ptr<outgoing_t>(d_arr, cudaFree);

	// allocate temporary host memory for table
	size_t wpitch = m_pitch / sizeof(outgoing_t);
	std::vector<outgoing_t> h_arr(height * wpitch, INVALID_OUTGOING);

	// allocate temporary host memory for row lengths
	std::vector<uint> h_rowLength(height, 0);

	// fill host memory
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {

		key_t key = i->first;
		const row_t& targets = i->second;

		assert(targets.size() <= wpitch);

		pidx_t partition = get<0>(key);
		nidx_t neuron = get<1>(key);

		size_t t_addr = outgoingRow(partition, neuron, wpitch);
		std::copy(targets.begin(), targets.end(), h_arr.begin() + t_addr);

		//! \todo move this into shared __device__/__host__ function
		size_t r_addr = partition * MAX_PARTITION_SIZE + neuron;
		h_rowLength.at(r_addr) = targets.size();
	}

	// delete accumulator memory which is no longer needed
	m_acc.clear();

	// copy table from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * m_pitch, cudaMemcpyHostToDevice));
	setOutgoingPitch(wpitch);

	// allocate device memory for row lengths
	uint* d_rowLength;
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_rowLength, height * sizeof(uint)));
	md_rowLength = boost::shared_ptr<uint>(d_rowLength, cudaFree);

	// copy row lengths from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_rowLength, &h_rowLength[0], h_rowLength.size() * sizeof(uint),
				cudaMemcpyHostToDevice));
}

