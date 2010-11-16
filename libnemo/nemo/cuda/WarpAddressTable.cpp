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
WarpAddressTable::addSynapse(
		const DeviceIdx& source,
		pidx_t targetPartition,
		delay_t delay1,
		size_t nextFreeWarp)
{
	row_key rk(source.partition, source.neuron, targetPartition, delay1);
	unsigned& rowSynapses = m_rowSynapses[rk];
	unsigned column = rowSynapses % WARP_SIZE;
	rowSynapses += 1;

	std::vector<size_t>& warps = m_warps[key(source.partition, source.neuron, delay1)][targetPartition];

	if(column == 0) {
		/* Add synapse to a new warp */
		warps.push_back(nextFreeWarp);
		m_warpsPerNeuronDelay[boost::make_tuple(source, delay1)] += 1;
		m_warpCount += 1;
		return SynapseAddress(nextFreeWarp, column);
	} else {
		/* Add synapse to an existing partially-filled warp */
		return SynapseAddress(*warps.rbegin(), column);
	}
}



void
WarpAddressTable::reportWarpSizeHistogram(std::ostream& out) const
{
	unsigned total = 0;
	std::vector<unsigned> hist(WARP_SIZE+1, 0);

	for(std::map<row_key, unsigned>::const_iterator i = m_rowSynapses.begin(); i != m_rowSynapses.end(); ++i) {
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
value_compare(const std::pair< boost::tuple<DeviceIdx, delay_t>, unsigned>& lhs,
		const std::pair< boost::tuple<DeviceIdx, delay_t>, unsigned>& rhs)
{
	return lhs.second < rhs.second;
}



unsigned
WarpAddressTable::warpsPerNeuronDelay(pidx_t p, nidx_t n, delay_t delay1) const
{
	typedef std::map< boost::tuple<DeviceIdx, delay_t>, unsigned>::const_iterator it;
	it i = m_warpsPerNeuronDelay.find(boost::make_tuple(DeviceIdx(p,n), delay1));
	if(i != m_warpsPerNeuronDelay.end()) {
		return i->second;
	} else {
		return 0;
	}
}



//! \todo can probably remove this
unsigned
WarpAddressTable::maxWarpsPerNeuronDelay() const
{
	if(m_warpsPerNeuronDelay.empty()) {
		return 0;
	}
	return std::max_element(
			m_warpsPerNeuronDelay.begin(), 
			m_warpsPerNeuronDelay.end(), value_compare)->second;
}


	} // end namespace cuda
} // end namespace nemo
