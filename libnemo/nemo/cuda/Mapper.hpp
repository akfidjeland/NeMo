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

#include <nemo/NetworkImpl.hpp>

#include "cuda_types.h"

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



/*! Maps between device and global indices */
class Mapper {

	public :

		Mapper(const nemo::NetworkImpl& net, unsigned partitionSize);

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

	private :

		unsigned m_partitionSize;

		unsigned m_partitionCount;

		unsigned m_offset;
};

	} // end namespace cuda
} // end namespace nemo

#endif
