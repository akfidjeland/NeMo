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

namespace nemo {
	namespace cuda {

/*! Neuron indices as used on CUDA devices
 *
 * The network is split into partitions when moved onto the device. Neurons on
 * the device are thus addressed using a two-level address. This class
 * encapsulates this and provides methods to convert between global addresses
 * and device addresses. 
 */
class DeviceIdx
{
	public:

		pidx_t partition;
		nidx_t neuron;

		DeviceIdx(nidx_t global) :
			partition(global / s_partitionSize),
			neuron(global % s_partitionSize) {}

		DeviceIdx(pidx_t p, nidx_t n) :
			partition(p),
			neuron(n) {}

		static void setPartitionSize(unsigned ps) { s_partitionSize = ps; }

		/*! \return the global address again */
		nidx_t hostIdx() const { return partition * s_partitionSize + neuron; }

	private:

		static unsigned s_partitionSize; // initialised in ConnectivityMatrix.cpp
};

	} // end namespace cuda
} // end namespace nemo

#endif
