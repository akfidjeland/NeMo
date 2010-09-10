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
	key idx(source.partition, source.neuron, targetPartition, delay);

	unsigned& rowSynapses = m_rowSynapses[idx];
	unsigned column = rowSynapses % WARP_SIZE;
	rowSynapses += 1;

	warp_set& warps = m_warps[idx];

	if(column == 0) {
		warps.insert(nextFreeWarp);
		m_warpsPerNeuron[source] += 1;
		m_warpCount += 1;
		return SynapseAddress(nextFreeWarp, column);
	} else {
		return SynapseAddress(*warps.rbegin(), column);
	}
}



unsigned
WarpAddressTable::maxWarpsPerNeuron() const
{
	if(m_warpsPerNeuron.empty()) {
		return 0;
	} else {
		return std::max_element(m_warpsPerNeuron.begin(), m_warpsPerNeuron.end())->second;
	}
}


	} // end namespace cuda
} // end namespace nemo
