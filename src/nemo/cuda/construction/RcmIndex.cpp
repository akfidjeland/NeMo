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
#include <nemo/cuda/rcm.cu_h>
#include <nemo/cuda/connectivityMatrix.cu_h>

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


RcmIndex::RcmIndex() :
	/* leave space for null warp at beginning */
	m_nextFreeWarp(1),
	m_data(WARP_SIZE, INVALID_REVERSE_SYNAPSE),
	m_forward(WARP_SIZE, 0)
{}



/*! Allocate space for a new RCM synapse for the given (target) neuron.
 *
 * \return
 * 		word offset for the synapse. This is the same for all the different
 * 		planes of data.
 */
size_t
RcmIndex::allocateSynapse(const DeviceIdx& target)
{
	key k(target.partition, target.neuron);
	unsigned& dataRowLength = m_dataRowLength[k];
	unsigned column = dataRowLength % WARP_SIZE;
	dataRowLength += 1;

	std::vector<size_t>& warps = m_warps[k];

	size_t row;
	if(column == 0) {
		/* Add synapse to a new warp */
		warps.push_back(m_nextFreeWarp);
		row = m_nextFreeWarp;
		m_nextFreeWarp += 1;
		/* Resize host buffers to accomodate the new warp. This allocation
		 * scheme could potentially result in a large number of reallocations,
		 * so we might be better off allocating larger chunks here */
		m_data.resize(m_nextFreeWarp * WARP_SIZE, INVALID_REVERSE_SYNAPSE);
		m_forward.resize(m_nextFreeWarp * WARP_SIZE, 0);
	} else {
		/* Add synapse to an existing partially-filled warp */
		row = *warps.rbegin();
	}
	return row * WARP_SIZE + column;
}



void
RcmIndex::addSynapse(
		const Synapse& s,
		const DeviceIdx& d_source,
		const DeviceIdx& d_target,
		size_t f_addr)
{
	size_t addr = allocateSynapse(d_target);
	m_data.at(addr) = r_packSynapse(d_source.partition, d_source.neuron, s.delay);
	m_forward.at(addr) = f_addr;
}


		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo
