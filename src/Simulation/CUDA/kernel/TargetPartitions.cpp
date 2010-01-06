#include "TargetPartitions.hpp"

#include <vector>
#include "boost/tuple/tuple_comparison.hpp"
#include <cuda_runtime.h>

#include "kernel.cu_h"
#include "util.h"


void 
TargetPartitions::addTargetPartition(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		pidx_t targetPartition)
{
	key_t key(sourcePartition, sourceNeuron, delay);
	m_acc[key].insert(targetPartition);
}



size_t
TargetPartitions::maxPitch() const
{
	size_t pitch = 0;
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		pitch = std::max(pitch, i->second.size());
	}
	return pitch;
}



void
TargetPartitions::moveToDevice(size_t partitionCount, size_t partitionSize)
{
	using namespace boost::tuples;

	size_t height = partitionCount * partitionSize * MAX_DELAY;
	size_t width = maxPitch() * sizeof(targetp_t);

	// allocate device memory
	targetp_t* d_arr;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)&d_arr, &m_pitch, width, height));
	md_arr = boost::shared_ptr<targetp_t>(d_arr, cudaFree);

	// allocate temporary host memory
	size_t wpitch = m_pitch / sizeof(targetp_t);
	std::vector<targetp_t> h_arr(height * wpitch, INVALID_PTARGET);

	// fill host memory
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {

		key_t key = i->first;
		const row_t& targets = i->second;

		assert(targets.size() <= wpitch);

		size_t addr = targetIdx(get<0>(key), get<1>(key), get<2>(key),
						partitionSize, wpitch);
		std::copy(targets.begin(), targets.end(), h_arr.begin() + addr);
	}

	// delete accumulator memory which is no longer needed
	m_acc.clear();

	// copy from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * m_pitch, cudaMemcpyHostToDevice));
	setTargetPitch(m_pitch);
}
