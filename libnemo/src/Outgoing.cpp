/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Outgoing.hpp"

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <boost/tuple/tuple_comparison.hpp>

#include "WarpAddressTable.hpp"
#include "kernel.cu_h"
#include "util.h"
#include "except.hpp"

namespace nemo {
	namespace cuda {

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



void
Outgoing::reportWarpSizeHistogram(std::ostream& out) const
{
	unsigned total = 0;
	std::vector<unsigned> hist(WARP_SIZE+1, 0);
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {
		targets_t targets = i->second;
		for(targets_t::const_iterator j = targets.begin(); j != targets.end(); ++j) {
			unsigned fullWarps = j->second / WARP_SIZE;
			unsigned partialWarp = j->second % WARP_SIZE;
			hist.at(WARP_SIZE) += fullWarps;
			total += fullWarps;
			if(partialWarp != 0) {
				hist.at(partialWarp) += 1;
				total += 1;
			}
		}
	}
	for(unsigned size=1; size < WARP_SIZE+1; ++size) {
		unsigned count = hist.at(size);
		double percentage = double(100 * count) / double(total);
		std::cout << size << ": " << count << "(" << percentage << "%)\n";
	}
	std::cout << "total: " << total << std::endl;
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
Outgoing::moveToDevice(size_t partitionCount, const WarpAddressTable& wtable)
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
	std::vector<unsigned> h_rowLength(height, 0);

	// accumulate the number of incoming warps for each partition.
	std::map<pidx_t, size_t> incoming;

	// fill host memory
	for(map_t::const_iterator i = m_acc.begin(); i != m_acc.end(); ++i) {

		skey_t key = i->first;
		const targets_t& targets = i->second;

		assert(targets.size() <= wpitch);

		pidx_t sourcePartition = get<0>(key);
		nidx_t sourceNeuron = get<1>(key);

		size_t t_addr = outgoingRow(sourcePartition, sourceNeuron, wpitch);

		size_t j = 0;
		for(targets_t::const_iterator r = targets.begin(); r != targets.end(); ++r) {

			pidx_t targetPartition = get<0>(r->first);
			delay_t delay = get<1>(r->first);
			//! \todo add run-time test that warp-size is as expected
			unsigned warps = DIV_CEIL(r->second, WARP_SIZE);

			incoming[targetPartition] += warps;

			//! \todo check for overflow here
			for(unsigned warp = 0; warp < warps; ++warp) {
				uint32_t offset = wtable.get(sourcePartition, sourceNeuron, targetPartition, delay);
				h_arr[t_addr + j + warp] =
					make_outgoing(targetPartition, delay, offset);
			}
			j += warps;
			assert(j <= wpitch);
		}

		//! \todo move this into shared __device__/__host__ function
		size_t r_addr = sourcePartition * MAX_PARTITION_SIZE + sourceNeuron;
		h_rowLength.at(r_addr) = warpCount(targets);
	}

	// delete accumulator memory which is no longer needed
	m_acc.clear();

	// copy table from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, &h_arr[0], height * m_pitch, cudaMemcpyHostToDevice));
	setOutgoingPitch(wpitch);

	// allocate device memory for row lengths
	unsigned* d_rowLength;
	err = cudaMalloc((void**)&d_rowLength, height * sizeof(unsigned));
	if(cudaSuccess != err) {
		throw DeviceAllocationException("outgoing spikes (row lengths)",
				height * sizeof(unsigned), err);
	}
	md_rowLength = boost::shared_ptr<unsigned>(d_rowLength, cudaFree);
	m_allocated += height * sizeof(unsigned);

	// copy row lengths from host to device
	CUDA_SAFE_CALL(cudaMemcpy(d_rowLength, &h_rowLength[0], h_rowLength.size() * sizeof(unsigned),
				cudaMemcpyHostToDevice));

	// return maximum number of incoming groups for any one partition
	return std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second;
}

	} // end namespace cuda
} // end namespace nemo
