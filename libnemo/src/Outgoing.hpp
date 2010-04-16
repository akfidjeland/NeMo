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

namespace nemo {
	namespace cuda {

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

		unsigned* count() const { return md_rowLength.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

		size_t totalWarpCount() const;

		/*! \return print histogram of sizes of each synapse
		 * warp to stdout */
		void reportWarpSizeHistogram(std::ostream& out) const;

		/*! Set the device data containing the outgoing spike groups.
		 *
		 * \return
		 * 		the maximum number of incoming warps for any one partition.
		 * 		This is a worst-case value, which assumes that every source
		 * 		neuron fires every cycle for some time.
		 */
		size_t moveToDevice(size_t partitionCount, const class WarpAddressTable& wtable);

	private :

		boost::shared_ptr<outgoing_t> md_arr;  // device data
		size_t m_pitch;                       // max pitch

		boost::shared_ptr<unsigned> md_rowLength; // per-neuron pitch

		typedef boost::tuple<pidx_t, delay_t> tkey_t;
		typedef std::map<tkey_t, unsigned> targets_t;

		typedef boost::tuple<pidx_t, nidx_t> skey_t;
		typedef std::map<skey_t, targets_t> map_t;

		map_t m_acc;

		size_t maxPitch() const;
		size_t warpCount(const targets_t& targets) const;

		size_t m_allocated;
};

	} // end namespace cuda
} // end namespace nemo

#endif
