#include "Outgoing.hpp"

#include <vector>
#include <cuda_runtime.h>
#include <boost/tuple/tuple_comparison.hpp>

#include "kernel.cu_h"
#include "util.h"
#include "except.hpp"

namespace nemo {

Outgoing::Outgoing() : m_allocated(0) {}



void 
Outgoing::addSynapse(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		pidx_t targetPartition)
{
	skey_t skey(sourcePartition, sourceNeuron);
	tkey_t tkey(targetPartition, delay);
	m_acc[skey][tkey] += 1;
}



size_t
Outgoing::warpCount(const targets_t& targets) const
{
	size_t warps = 0;
	for(targets_t::const_iterator i = targets.begin(); i != targets.end(); ++i) {
		warps += DIV_CEIL(i->second, WARP_SIZE);
	}
	return warps;
}



size_t
Outgoing::totalWarpCount() const
{
	size_t count = 0;
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		count += warpCount(i->second);
	}
	return count;
}



size_t
Outgoing::maxPitch() const
{
	size_t pitch = 0;
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		pitch = std::max(pitch, warpCount(i->second));
	}
	return pitch;
}



bool
compare_warp_counts(
		const std::pair<pidx_t, size_t>& lhs,
		const std::pair<pidx_t, size_t>& rhs)
{
	return lhs.second < rhs.second;
}



size_t
Outgoing::moveToDevice(size_t partitionCount,
				const std::map<fcm_key_t, SynapseGroup>& fcm)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE;
	size_t width = maxPitch() * sizeof(outgoing_t);

	// allocate device memory for table
	outgoing_t* d_arr;
	cudaError err = cudaMallocPitch((void**)&d_arr, &m_pitch, width, height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("outgoing spikes", width * height, err);
	}
	md_arr = boost::shared_ptr<outgoing_t>(d_arr, cudaFree);

	m_allocated = m_pitch * height;

	// allocate temporary host memory for table
	size_t wpitch = m_pitch / sizeof(outgoing_t);
	std::vector<outgoing_t> h_arr(height * wpitch, INVALID_OUTGOING);

	// allocate temporary host memory for row lengths
	std::vector<uint> h_rowLength(height, 0);

	// accumulate the number of incoming warps for each partition.
	std::map<pidx_t, size_t> incoming;

	// fill host memory
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {

		skey_t key = i->first;
		const targets_t& targets = i->second;

		assert(targets.size() <= wpitch);

		//! \todo rename sourcePN
		pidx_t partition = get<0>(key);
		nidx_t neuron = get<1>(key);

		size_t t_addr = outgoingRow(partition, neuron, wpitch);

		size_t j = 0;
		for(targets_t::const_iterator r = targets.begin(); r != targets.end(); ++r) {
			tkey_t tkey = r->first;
			pidx_t targetPartition = get<0>(tkey);
			delay_t delay = get<1>(tkey);
			//! \todo add run-time test that warp-size is as expected
			uint warps = DIV_CEIL(r->second, WARP_SIZE);

			std::map<fcm_key_t, SynapseGroup>::const_iterator groupref =
					fcm.find(fcm_key_t(partition, targetPartition, delay));
			assert(groupref != fcm.end());

			incoming[targetPartition] += warps;

			//! \todo check for overflow here
			for(uint warp = 0; warp < warps; ++warp) {
				uint32_t offset = groupref->second.warpOffset(neuron, warp);
				h_arr[t_addr + j + warp] =
					make_outgoing(targetPartition, delay, offset);
			}
			j += warps;
			assert(j <= wpitch);
		}

		//! \todo move this into shared __device__/__host__ function
		size_t r_addr = partition * MAX_PARTITION_SIZE + neuron;
		h_rowLength.at(r_addr) = warpCount(targets);
	}

	// delete accumulator memory which is no longer needed
	m_acc.clear();

	// copy table from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * m_pitch, cudaMemcpyHostToDevice));
	setOutgoingPitch(wpitch);

	// allocate device memory for row lengths
	uint* d_rowLength;
	err = cudaMalloc((void**)&d_rowLength, height * sizeof(uint));
	if(cudaSuccess != err) {
		throw DeviceAllocationException("outgoing spikes (row lengths)",
				height * sizeof(uint), err);
	}
	md_rowLength = boost::shared_ptr<uint>(d_rowLength, cudaFree);
	m_allocated += height * sizeof(uint);

	// copy row lengths from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_rowLength, &h_rowLength[0], h_rowLength.size() * sizeof(uint),
				cudaMemcpyHostToDevice));

	// return maximum number of incoming groups for any one partition
	return std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second;
}

} // end namespace nemo
