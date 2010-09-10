/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "WarpAddressTable.hpp"
#include <boost/tuple/tuple_comparison.hpp>

#include <nemo/exception.hpp>
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {


WarpAddressTable::WarpAddressTable() :
	m_warpCount(0)
{
	;
}


SynapseAddress
WarpAddressTable::addSynapse(const DeviceIdx& source, pidx_t targetPartition, delay_t delay, size_t nextFreeWarp)
{
	idx_t idx(source.partition, source.neuron, targetPartition, delay);

	unsigned& rowSynapses = m_rowSynapses[idx];
	unsigned column = rowSynapses % WARP_SIZE;
	rowSynapses += 1;

	std::set<size_t>& warps = m_warps[idx];

	if(column == 0) {
		warps.insert(nextFreeWarp);
		m_warpsPerNeuron[source] += 1;
		m_warpCount += 1;
		return SynapseAddress(nextFreeWarp, column);
	} else {
		return SynapseAddress(*warps.rbegin(), column);
	}
}


#if 1
size_t
WarpAddressTable::get(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	idx_t idx(sp, sn, tp, d);
	warp_map::const_iterator wa = m_warps.find(idx);
	if(wa == m_warps.end()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "invalid row in WarpAddressTable lookup");
	}
	return *wa->second.begin();
}
#endif



const WarpAddressTable::warp_set&
WarpAddressTable::warpSet(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	idx_t idx(sp, sn, tp, d);
	warp_map::const_iterator wa = m_warps.find(idx);
	if(wa == m_warps.end()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "invalid row in WarpAddressTable lookup");
	}
	return wa->second;
}



WarpAddressTable::const_iterator
WarpAddressTable::warps_begin(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	return warpSet(sp, sn, tp, d).begin();
}



WarpAddressTable::const_iterator
WarpAddressTable::warps_end(pidx_t sp, nidx_t sn, pidx_t tp, delay_t d) const
{
	return warpSet(sp, sn, tp, d).end();
}



unsigned
WarpAddressTable::maxWarpsPerNeuron() const
{
	return std::max_element(m_warpsPerNeuron.begin(), m_warpsPerNeuron.end())->second;
}


	} // end namespace cuda
} // end namespace nemo
