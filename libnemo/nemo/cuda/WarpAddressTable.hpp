#ifndef WARP_ADDRESS_TABLE_HPP
#define WARP_ADDRESS_TABLE_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <set>

#include <boost/tuple/tuple.hpp>

#include <nemo/types.hpp>
//! \todo move DeviceIdx to types.hpp
#include "Mapper.hpp"
#include "types.h"

namespace nemo {
	namespace cuda {

class WarpAddressTable
{
	public :

		/* Synapses are grouped into 'rows' which share the same source neuron,
		 * target partition
		 *                   source  source  target */
		typedef boost::tuple<pidx_t, nidx_t, pidx_t, delay_t> key;

		/* Each row may be spread over a disparate set of warps */
		typedef std::set<size_t> warp_set;

	private :

		typedef std::map<key, warp_set> warp_map;

	public :

		WarpAddressTable();

		/*
		 * \param nextFreeWarp
		 * 		The next unused warp in the host FCM.
		 *
		 * \return
		 * 		Address of this synapse in the form of a warp address and a
		 * 		within-warp address. This might refer to an existing warp or a
		 * 		new warp.
		 */
		SynapseAddress addSynapse(const DeviceIdx&, pidx_t, delay_t, size_t nextFreeWarp);

		typedef warp_map::const_iterator row_iterator;

		row_iterator row_begin() const { return m_warps.begin(); }
		row_iterator row_end() const { return m_warps.end(); }

		unsigned warpCount() const { return m_warpCount; }

		unsigned maxWarpsPerNeuron() const;

	private :

		warp_map m_warps;

		std::map<key, unsigned> m_rowSynapses;

		std::map<DeviceIdx, unsigned> m_warpsPerNeuron;

		unsigned m_warpCount;
};

	} // end namespace cuda
} // end namespace nemo

#endif
