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

#include <nemo/util.h>

#include "WarpAddressTable.hpp"
#include "device_memory.hpp"
#include "exception.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {

Outgoing::Outgoing() : m_pitch(0), m_allocated(0), m_maxIncomingWarps(0) {}


Outgoing::Outgoing(size_t partitionCount, const WarpAddressTable& wtable) :
		m_pitch(0),
		m_allocated(0),
		m_maxIncomingWarps(0)
{
	init(partitionCount, wtable);
}



bool
compare_warp_counts(
		const std::pair<pidx_t, size_t>& lhs,
		const std::pair<pidx_t, size_t>& rhs)
{
	return lhs.second < rhs.second;
}


void
Outgoing::init(size_t partitionCount, const WarpAddressTable& wtable)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE * MAX_DELAY;
	size_t width = wtable.maxWarpsPerNeuronDelay() * sizeof(outgoing_t);

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

		pidx_t sourcePartition = get<0>(k);
		nidx_t sourceNeuron = get<1>(k);
		pidx_t targetPartition = get<2>(k);
		delay_t delay1 = get<3>(k);

		const WarpAddressTable::warp_set& r = ri->second;

		typedef WarpAddressTable::warp_set::const_iterator warp_iterator;

		size_t r_addr = outgoingCountOffset(sourcePartition, sourceNeuron, delay1-1);

		for(warp_iterator wi = r.begin(); wi != r.end(); ++wi) {
			//! \todo use DeviceIdx overload here. Refactor to share with r_addr
			size_t rowBegin = outgoingRow(sourcePartition, sourceNeuron, delay1-1, wpitch);
			//! \todo can increment this in one go outside loop
			size_t col = h_rowLength[r_addr]++;
			h_arr[rowBegin + col] = make_outgoing(targetPartition, *wi);
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
	m_maxIncomingWarps = incoming.size() ? std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second : 0;
}


	} // end namespace cuda
} // end namespace nemo
