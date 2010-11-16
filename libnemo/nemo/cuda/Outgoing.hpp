#ifndef NEMO_CUDA_OUTGOING_HPP
#define NEMO_CUDA_OUTGOING_HPP

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

#include "outgoing.cu_h"

namespace nemo {
	namespace cuda {

class Outgoing
{
	public :

		Outgoing();

		/*! Set the device data containing the outgoing spike groups. */
		Outgoing(size_t partitionCount, const class WarpAddressTable& wtable);

		/*! \return device pointer to the outgoing data */
		outgoing_t* d_data() const { return md_arr.get(); }

		/*! \return device pointer to address table */
		outgoing_addr_t* d_addr() const { return md_rowLength.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

		/* \return
		 * 		the maximum number of incoming warps for any one partition.
		 * 		This is a worst-case value, which assumes that every source
		 * 		neuron fires every cycle for some time. */
		size_t maxIncomingWarps() const { return m_maxIncomingWarps; }

	private :

		void init(size_t partitionCount, const class WarpAddressTable& wtable);

		boost::shared_ptr<outgoing_t> md_arr; // device data
		size_t m_pitch;                       // max pitch

		/* Store offset/length pairs here in order to address arbitrary entries
		 * in md_arr where the entries have variable width (for compactness).
		 * md_rowLength have fixed-width entries so addressing is
		 * straightforward (based on a neuron/delay pair) */
		boost::shared_ptr<outgoing_addr_t> md_rowLength; // per neuron/delay pitch

		size_t m_allocated;

		size_t m_maxIncomingWarps;
};

	} // end namespace cuda
} // end namespace nemo

#endif
