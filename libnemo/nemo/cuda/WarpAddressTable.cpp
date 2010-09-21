/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/tuple/tuple_comparison.hpp>

#include "WarpAddressTable.hpp"
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



void
WarpAddressTable::reportWarpSizeHistogram(std::ostream& out) const
{
	unsigned total = 0;
	std::vector<unsigned> hist(WARP_SIZE+1, 0);

	for(std::map<key, unsigned>::const_iterator i = m_rowSynapses.begin(); i != m_rowSynapses.end(); ++i) {
		unsigned fullWarps = i->second / WARP_SIZE;
		unsigned partialWarp = i->second % WARP_SIZE;
		hist.at(WARP_SIZE) += fullWarps;
		total += fullWarps;
		if(partialWarp != 0) {
			hist.at(partialWarp) += 1;
			total += 1;
		}
	}
	for(unsigned size=1; size < WARP_SIZE+1; ++size) {
		unsigned count = hist.at(size);
		double percentage = double(100 * count) / double(total);
		out << size << ": " << count << "(" << percentage << "%)\n";
	}
	out << "total: " << total << std::endl;
}


bool
value_compare(const std::pair<DeviceIdx,
		unsigned>& lhs, const std::pair<DeviceIdx, unsigned>& rhs)
{
	return lhs.second < rhs.second;
}


unsigned
WarpAddressTable::maxWarpsPerNeuron() const
{
	if(m_warpsPerNeuron.empty()) {
		return 0;
	}
	return std::max_element(m_warpsPerNeuron.begin(), m_warpsPerNeuron.end(), value_compare)->second;
}


	} // end namespace cuda
} // end namespace nemo
