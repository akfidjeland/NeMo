#ifndef NEMO_CUDA_RCM_INDEX_HPP
#define NEMO_CUDA_RCM_INDEX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

#include <nemo/types.hpp>
//! \todo move DeviceIdx to types.hpp
#include <nemo/cuda/Mapper.hpp>
#include <nemo/cuda/types.h>


namespace nemo {
	namespace cuda {
		namespace construction {

class RcmIndex
{
	public :

		typedef boost::tuple<pidx_t, nidx_t> key;

		/*
		 * \param nextFreeWarp
		 * 		The next unused warp in the host FCM.
		 *
		 * \return
		 * 		Address of this synapse in the form of a warp address and a
		 * 		within-warp address. This might refer to an existing warp or a
		 * 		new warp.
		 */
		SynapseAddress addSynapse(const DeviceIdx& target, size_t nextFreeWarp);

	private :

		typedef boost::unordered_map<key, std::vector<size_t> > warp_map;

		warp_map m_warps;

		/* In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		boost::unordered_map<key, unsigned> m_dataRowLength;
};



		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo


#endif
