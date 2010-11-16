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
#include <vector>

#include <boost/tuple/tuple.hpp>

#include <nemo/types.hpp>
//! \todo move DeviceIdx to types.hpp
#include "Mapper.hpp"
#include "types.h"
#include "outgoing.cu_h"

namespace nemo {
	namespace cuda {

/*! \brief Mapping from source-neuron/target-neuron/delay to synapse warp addresses
 *
 * This mapping is a temporary structure used during construction of the device
 * data for the forward connectivity matrix. Synapses are organised in groups,
 * where each group contains synapses sharing the same source neuron, target
 * neuron and delay. Each group is in turn split into fixed-size warps, which
 * is the basic unit for spike delivery in NeMo.
 *
 * The warp address table serves two purposes:
 *
 * 1. allocating addresses (warp index + column index) for each synapse
 * 2. accumulate the addressing data that will be used at run-time.
 *
 * The warp address table is intended for incremental construction, by adding
 * individual synapses.
 *
 * At run-time this mapping is found in the data structure \a Outgoing.
 *
 * \see Outgoing
 */
class WarpAddressTable
{
	public :

		/* Synapses are grouped into 'rows' which share the same source neuron,
		 * target partition and delay
		 *                   source  source  target */
		typedef boost::tuple<pidx_t, nidx_t, pidx_t, delay_t> row_key;

		/* However, at run-time we get addresses only based on the source
		 * neuron and delay. */
		typedef boost::tuple<pidx_t, nidx_t, delay_t> key;

		/* Each row may be spread over a disparate set of warps. Each target
		 * partition may have synapses in several warps. */
		//! \todo consider using unordered here instead (for faster lookup).
		typedef std::map<pidx_t, std::vector<size_t> > row_t;

	private :

		typedef std::map<key, row_t> warp_map;

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

		typedef warp_map::const_iterator iterator;

		iterator begin() const { return m_warps.begin(); }
		iterator end() const { return m_warps.end(); }

		typedef row_t::const_iterator row_iterator;

		unsigned maxWarpsPerNeuronDelay() const;

		/*! \return print histogram of sizes of each synapse
		 * warp to stdout */
		void reportWarpSizeHistogram(std::ostream& out) const;

	private :

		warp_map m_warps;

		/* In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		std::map<row_key, unsigned> m_rowSynapses;

		//! \todo can get rid of this one now
		std::map< boost::tuple<DeviceIdx, delay_t>, unsigned> m_warpsPerNeuronDelay;

		unsigned m_warpCount;
};

	} // end namespace cuda
} // end namespace nemo

#endif
