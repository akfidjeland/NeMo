/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <iostream>

#include <boost/tuple/tuple_comparison.hpp>

#include <nemo/cuda/kernel.cu_h>

#include "RcmIndex.hpp"


namespace boost {
	namespace tuples {


//! \todo share the hashing code with FcmIndex
template<typename T1, typename T2>
std::size_t
hash_value(const tuple<T1, T2>& k)
{
	std::size_t seed = 0;
	boost::hash_combine(seed, boost::tuples::get<0>(k));
	boost::hash_combine(seed, boost::tuples::get<1>(k));
	return seed;
}

	} // end namespace tuples
} // end namespace boost


namespace nemo {
	namespace cuda {
		namespace construction {



SynapseAddress
RcmIndex::addSynapse(
		const DeviceIdx& target,
		size_t nextFreeWarp)
{
	// data_key dk(source.partition, source.neuron, targetPartition, delay1);
	key k(target.partition, target.neuron);
	unsigned& dataRowLength = m_dataRowLength[k];
	unsigned column = dataRowLength % WARP_SIZE;
	dataRowLength += 1;

	std::vector<size_t>& warps = m_warps[k];

	if(column == 0) {
		/* Add synapse to a new warp */
		warps.push_back(nextFreeWarp);
		return SynapseAddress(nextFreeWarp, column);
	} else {
		/* Add synapse to an existing partially-filled warp */
		return SynapseAddress(*warps.rbegin(), column);
	}
}


		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo
