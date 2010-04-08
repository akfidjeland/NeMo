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
#include <boost/tuple/tuple.hpp>
#include "nemo_cuda_types.h"

namespace nemo {
	namespace cuda {

class WarpAddressTable
{
	public :

		void set(pidx_t, nidx_t, pidx_t, delay_t, size_t);

		size_t get(pidx_t, nidx_t, pidx_t, delay_t) const;

	private :

		//                   source  source  target
		typedef boost::tuple<pidx_t, nidx_t, pidx_t, delay_t> idx_t;
		std::map<idx_t, size_t> m_data;
};

	} // end namespace cuda
} // end namespace nemo

#endif
