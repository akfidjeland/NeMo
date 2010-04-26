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

		explicit Mapper(unsigned partitionSize) :
			m_partitionSize(partitionSize) {}

		DeviceIdx deviceIdx(nidx_t global) const {
			return DeviceIdx(global / m_partitionSize, global % m_partitionSize);
		}

		nidx_t hostIdx(DeviceIdx d) const {
			return d.partition * m_partitionSize + d.neuron;
		}

		nidx_t hostIdx(pidx_t p, nidx_t n) const {
			return p * m_partitionSize + n;
		}

	private :

		unsigned m_partitionSize;
};

	} // end namespace cuda
} // end namespace nemo

#endif
