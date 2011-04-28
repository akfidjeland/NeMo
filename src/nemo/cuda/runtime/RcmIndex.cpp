/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/tuple/tuple.hpp>

#include <nemo/util.h>
#include <nemo/cuda/runtime/RcmIndex.hpp>
#include <nemo/cuda/construction/RcmIndex.hpp>
#include <nemo/cuda/device_memory.hpp>
#include <nemo/cuda/parameters.cu_h>
#include <nemo/cuda/rcm.cu_h>
#include <nemo/cuda/kernel.cu_h>

namespace nemo {
	namespace cuda {
		namespace runtime {


inline
rcm_index_address_t
make_rcm_index_address(uint start, uint len)
{
	return make_uint2(start, len);
}



RcmIndex::RcmIndex(size_t partitionCount, const construction::RcmIndex& index):
	m_pitch(0),
	m_allocated(0)
{
	using namespace boost::tuples;

	const size_t maxNeuronCount = partitionCount * MAX_PARTITION_SIZE;
	std::vector<rcm_index_address_t> h_address(maxNeuronCount, INVALID_RCM_INDEX_ADDRESS);
	std::vector<rcm_address_t> h_data;

	/* Populate the host-side data structures */

	typedef construction::RcmIndex::warp_map::const_iterator iterator;
	typedef construction::RcmIndex::key key;

	size_t allocated = 0; // words, so far

	for(iterator i = index.m_warps.begin(); i != index.m_warps.end(); ++i) {

		const key k = i->first;
		const unsigned targetPartition = get<0>(k);
		const unsigned targetNeuron = get<1>(k);
		const std::vector<size_t>& row = i->second;

		h_address.at(rcm_metaIndexAddress(targetPartition, targetNeuron)) =
			make_rcm_index_address(allocated, row.size());

		/* Each row in the index is padded out to the next warp boundary */
		size_t nWarps = row.size();
		size_t nWords = ALIGN(nWarps, WARP_SIZE);
		size_t nPadding = nWords - nWarps;

		std::copy(row.begin(), row.end(), std::back_inserter(h_data));          // data
		std::fill_n(std::back_inserter(h_data), nPadding, INVALID_RCM_ADDRESS); // padding

		allocated += nWords;
	}

	/* Copy index addresses to device */
	if(!h_address.empty()) {
		md_address = d_array<rcm_index_address_t>(h_address.size(), "RCM index addresses");
		memcpyToDevice(md_address.get(), h_address);
		m_allocated += h_address.size() * sizeof(rcm_index_address_t);
	}

	/* Copy index data to device */
	if(allocated != 0) {
		md_data = d_array<rcm_address_t>(allocated, "RCM index data");
		memcpyToDevice(md_data.get(), h_data);
		m_allocated += allocated * sizeof(rcm_address_t);
	}
}



		}
	}
}
