/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Outgoing.hpp"

#include <vector>
#include <cuda_runtime.h>
#include <boost/tuple/tuple_comparison.hpp>

#include <nemo/util.h>

#include "WarpAddressTable.hpp"
#include "device_memory.hpp"
#include "exception.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {

Outgoing::Outgoing() : m_pitch(0), m_allocated(0) {}




void
Outgoing::reportWarpSizeHistogram(std::ostream& out) const
{
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Warp size reporting currently disabled");
#if 0
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
		out << size << ": " << count << "(" << percentage << "%)\n";
	}
	out << "total: " << total << std::endl;
#endif
}



bool
compare_warp_counts(
		const std::pair<pidx_t, size_t>& lhs,
		const std::pair<pidx_t, size_t>& rhs)
{
	return lhs.second < rhs.second;
}



//! \todo call directly from ctor
size_t
Outgoing::moveToDevice(size_t partitionCount, const WarpAddressTable& wtable)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE;
	size_t width = wtable.maxWarpsPerNeuron() * sizeof(outgoing_t);

	// allocate device memory for table
	outgoing_t* d_arr = NULL;
	d_mallocPitch((void**)&d_arr, &m_pitch, width, height, "outgoing spikes");
	md_arr = boost::shared_ptr<outgoing_t>(d_arr, d_free);

	m_allocated = m_pitch * height;

	// allocate temporary host memory for table
	size_t wpitch = m_pitch / sizeof(outgoing_t);
	std::vector<outgoing_t> h_arr(height * wpitch, INVALID_OUTGOING);

	// allocate temporary host memory for row lengths
	std::vector<unsigned> h_rowLength(height, 0);

	// accumulate the number of incoming warps for each partition.
	std::map<pidx_t, size_t> incoming;

	// fill host memory
	for(WarpAddressTable::row_iterator ri = wtable.row_begin(); ri != wtable.row_end(); ++ri) {

		const WarpAddressTable::key& k = ri->first;

		//! \todo store DeviceIdx directly
		pidx_t sourcePartition = get<0>(k);
		nidx_t sourceNeuron = get<1>(k);
		pidx_t targetPartition = get<2>(k);
		delay_t delay = get<3>(k);

		const WarpAddressTable::warp_set& r = ri->second;

		typedef WarpAddressTable::warp_set::const_iterator warp_iterator;

		//! \todo move this into shared __device__/__host__ function
		size_t r_addr = sourcePartition * MAX_PARTITION_SIZE + sourceNeuron;

		for(warp_iterator wi = r.begin(); wi != r.end(); ++wi) {
			//! \todo use DeviceIdx overload here. Refactor to share with r_addr
			size_t rowBegin = outgoingRow(sourcePartition, sourceNeuron, wpitch);
			size_t col = h_rowLength[r_addr]++;
			h_arr[rowBegin + col] = make_outgoing(targetPartition, delay, *wi);
			incoming[targetPartition] += 1;
		}
	}

	// copy table from host to device
	if(d_arr != NULL && !h_arr.empty()) {
		memcpyToDevice(d_arr, h_arr, height * wpitch);
	}
	CUDA_SAFE_CALL(setOutgoingPitch(wpitch));

	// allocate device memory for row lengths
	unsigned* d_rowLength = NULL;
	d_malloc((void**)&d_rowLength, height * sizeof(unsigned), "outgoing spikes (row lengths)");
	md_rowLength = boost::shared_ptr<unsigned>(d_rowLength, d_free);
	m_allocated += height * sizeof(unsigned);

	memcpyToDevice(d_rowLength, h_rowLength);

	// return maximum number of incoming groups for any one partition
	//! \todo compute this on forward pass
	return incoming.size() ? std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second : 0;
}

	} // end namespace cuda
} // end namespace nemo
