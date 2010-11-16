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

	if(width == 0) {
		/* No synapses, so nothing to do here */
		return;
	}

	// allocate device memory for table
	outgoing_t* d_arr = NULL;
	d_mallocPitch((void**)&d_arr, &m_pitch, width, height, "outgoing spikes");
	md_arr = boost::shared_ptr<outgoing_t>(d_arr, d_free);

	m_allocated = m_pitch * height;

	// allocate temporary host memory for table
	size_t wpitch = m_pitch / sizeof(outgoing_t);
	std::vector<outgoing_t> h_arr(height * wpitch, INVALID_OUTGOING);

	// allocate temporary host memory for row lengths
	std::vector<outgoing_addr_t> h_addr(height, make_outgoing(0,0));

	// accumulate the number of incoming warps for each partition.
	std::map<pidx_t, size_t> incoming;

	// fill host memory
	for(WarpAddressTable::iterator ti = wtable.begin(); ti != wtable.end(); ++ti) {

		const WarpAddressTable::key& k = ti->first;

		pidx_t sourcePartition = get<0>(k);
		nidx_t sourceNeuron = get<1>(k);
		delay_t delay1 = get<2>(k);

		//! \todo use DeviceIdx overload here. Refactor to share with r_addr
		size_t rowBegin = outgoingRow(sourcePartition, sourceNeuron, delay1-1, wpitch);
		unsigned col = 0;

		/* iterate over target partitions in a row */
		const WarpAddressTable::row_t& r = ti->second;
		for(WarpAddressTable::row_iterator ri = r.begin(); ri != r.end(); ++ri) {

			pidx_t targetPartition = ri->first;
			const std::vector<size_t>& warps = ri->second;
			size_t len = warps.size();
			incoming[targetPartition] += len;
			outgoing_t* p = &h_arr[rowBegin + col];
			col += len;

			/* iterate over warps specific to a target partition */
			for(std::vector<size_t>::const_iterator wi = warps.begin();
					wi != warps.end(); ++wi, ++p) {
				*p = make_outgoing(targetPartition, *wi);
			}
		}

		/* Set address info here, since both start and length are now known */
		size_t r_addr = outgoingAddrOffset(sourcePartition, sourceNeuron, delay1-1);
		h_addr[r_addr] = make_outgoing_addr(rowBegin, col);
	}

	/* copy table from host to device */
	if(d_arr != NULL && !h_arr.empty()) {
		memcpyToDevice(d_arr, h_arr, height * wpitch);
	}

	CUDA_SAFE_CALL(setOutgoingPitch(wpitch));

	/* scatterLocal assumes that wpitch <= THREADS_PER_BLOCK. It would possible
	 * to write this in order to handle the other case as well, with different
	 * looping logic. Separate kernels might be more sensible. */
	if(wpitch > THREADS_PER_BLOCK) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Outgoing pitch too wide");
	}
	CUDA_SAFE_CALL(setOutgoingStep(THREADS_PER_BLOCK / wpitch));

	// allocate device memory for row lengths
	outgoing_addr_t* d_addr = NULL;
	d_malloc((void**)&d_addr, height * sizeof(outgoing_addr_t), "outgoing spikes (row lengths)");
	md_rowLength = boost::shared_ptr<outgoing_addr_t>(d_addr, d_free);
	m_allocated += height * sizeof(outgoing_addr_t);

	memcpyToDevice(d_addr, h_addr);

	// return maximum number of incoming groups for any one partition
	//! \todo compute this on forward pass
	m_maxIncomingWarps = incoming.size() ? std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second : 0;
}


	} // end namespace cuda
} // end namespace nemo
