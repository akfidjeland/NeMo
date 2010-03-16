#ifndef OUTGOING_HPP
#define OUTGOING_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include "outgoing.cu_h"
#include "SynapseGroup.hpp"

namespace nemo {

class Outgoing
{
	public :

		Outgoing();

		void addSynapse(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition);


		outgoing_t* data() const { return md_arr.get(); }

		uint* count() const { return md_rowLength.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

		size_t totalWarpCount() const;

		/*! \return print histogram of sizes of each synapse
		 * warp to stdout */
		void reportWarpSizeHistogram(std::ostream& out) const;

	private :

		typedef boost::tuple<pidx_t, pidx_t, delay_t> fcm_key_t; // source, target, delay

	public :

		/*! Set the device data containing the outgoing spike groups.
		 *
		 * \return
		 * 		the maximum number of incoming warps for any one partition.
		 * 		This is a worst-case value, which assumes that every source
		 * 		neuron fires every cycle for some time.
		 */
		size_t moveToDevice(size_t partitionCount,
				const std::map<fcm_key_t, class SynapseGroup>& fcm);

	private :

		boost::shared_ptr<outgoing_t> md_arr;  // device data
		size_t m_pitch;                       // max pitch

		boost::shared_ptr<uint> md_rowLength; // per-neuron pitch

		typedef boost::tuple<pidx_t, delay_t> tkey_t;
		typedef std::map<tkey_t, uint> targets_t;

		typedef boost::tuple<pidx_t, nidx_t> skey_t;
		typedef std::map<skey_t, targets_t> map_t;

		map_t m_acc;

		size_t maxPitch() const;
		size_t warpCount(const targets_t& targets) const;

		size_t m_allocated;
};

} // end namespace nemo

#endif
