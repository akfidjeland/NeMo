/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "WarpAddressTable.hpp"

#include <assert.h>
#include <stdexcept>
#include <boost/tuple/tuple_comparison.hpp>

namespace nemo {
	namespace cuda {

void
WarpAddressTable::set(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d, size_t wa)
{
	idx_t idx(sp, sn, tp, d);
	if(m_data.find(idx) != m_data.end()) {
		throw std::logic_error("Warp address table entry set twice");
	}
	m_data[idx] = wa;
}



size_t
WarpAddressTable::get(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	idx_t idx(sp, sn, tp, d);
	std::map<idx_t, size_t>::const_iterator wa = m_data.find(idx);
	if(wa == m_data.end()) {
		throw std::out_of_range("invalid neuron in WarpAddressTable lookup");
	}
	return wa->second;
}

	} // end namespace cuda
} // end namespace nemo
