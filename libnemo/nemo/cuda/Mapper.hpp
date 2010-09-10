#ifndef NEMO_CUDA_DEVICE_IDX_HPP
#define NEMO_CUDA_DEVICE_IDX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <set>
#include <nemo/NetworkImpl.hpp>

#include "types.h"

namespace nemo {
	namespace cuda {

/*! Neuron indices as used on CUDA devices
 *
 * The network is split into partitions when moved onto the device. Neurons on
 * the device are thus addressed using a two-level address. */
struct DeviceIdx
{
	public:

		pidx_t partition;
		nidx_t neuron;

		DeviceIdx(pidx_t p, nidx_t n) : partition(p), neuron(n) {}
};


inline
bool
operator<(const DeviceIdx& lhs, const DeviceIdx& rhs)
{
	return lhs.partition < rhs.partition ||
		(lhs.partition == rhs.partition && lhs.neuron < rhs.neuron);
}



/*! Maps between device and global indices */
class Mapper {

	public :

		Mapper(const nemo::network::NetworkImpl& net, unsigned partitionSize);

		/* Add global neuron index to the set of 'valid' synapses and return
		 * the correspondong device neuron index */
		DeviceIdx addIdx(nidx_t global);

		DeviceIdx deviceIdx(nidx_t global) const;

		nidx_t hostIdx(DeviceIdx d) const {
			return m_offset + d.partition * m_partitionSize + d.neuron;
		}

		nidx_t hostIdx(pidx_t p, nidx_t n) const {
			return m_offset + p * m_partitionSize + n;
		}

		unsigned partitionSize() const { return m_partitionSize; }

		unsigned partitionCount() const { return m_partitionCount; }

		unsigned minHostIdx() const { return m_offset; }

		unsigned maxHostIdx() const;

		bool valid(nidx_t global) const { return m_validGlobal.count(global) == 1; }

	private :

		unsigned m_partitionSize;

		unsigned m_partitionCount;

		unsigned m_offset;

		std::set<nidx_t> m_validGlobal;
};

	} // end namespace cuda
} // end namespace nemo

#endif
