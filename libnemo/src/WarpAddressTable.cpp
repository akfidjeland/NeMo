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
#include <boost/tuple/tuple_comparison.hpp>

namespace nemo {

void
WarpAddressTable::set(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d, size_t wa)
{
	idx_t idx(sp, sn, tp, d);
	// we should only set each datum once
	assert(m_data.find(idx) == m_data.end());
	m_data[idx] = wa;
}



size_t
WarpAddressTable::get(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	idx_t idx(sp, sn, tp, d);
	std::map<idx_t, size_t>::const_iterator wa = m_data.find(idx);
	//! \todo throw exception here instead
	assert(wa != m_data.end());
	return wa->second;
}

} // end namespace nemo
