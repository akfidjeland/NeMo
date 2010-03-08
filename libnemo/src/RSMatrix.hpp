#ifndef RS_MATRIX_HPP
#define RS_MATRIX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file RSMatrix.hpp

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>

#include "kernel.cu_h"
#include "nemo_cuda_types.h"

namespace nemo {

/*! \brief Sparse synapse matrix in reverse format for a single partition
 *
 * Synapses in this matrix are stored on a per-target basis.
 *
 * The reverse matrix has two planes: one for reverse addressing and one for
 * accumulating STDP statistics (LTP and LTD).
 *
 * \see SMatrix
 *
 * \author Andreas Fidjeland
 */

struct RSMatrix
{
	public:

		RSMatrix(size_t partitionSize);

		void addSynapse(
				unsigned int sourcePartition,
				unsigned int sourceNeuron,
				unsigned int sourceSynapse,
				unsigned int targetNeuron,
				unsigned int delay);

	private :

		typedef boost::tuple<pidx_t, pidx_t, delay_t> fcm_key_t; // source, target, delay

	public :

		void moveToDevice(
				const std::map<fcm_key_t, class SynapseGroup>& fcm,
				pidx_t targetPartition);

		void clearStdpAccumulator();

		/*! \return bytes allocated on the device */
		size_t d_allocated() const;

		/*! \return word pitch, i.e. max number of synapses per neuron */
		size_t pitch() const { return m_pitch; }

		/*! \return device address of reverse address matrix */
		uint32_t* d_address() const;

		/*! \return device address of STDP accumulator matrix */
		float* d_stdp() const;

		uint32_t* d_faddress() const;

	private:

		boost::shared_ptr<uint32_t> m_deviceData;

		typedef std::vector< std::vector<uint32_t> > host_sparse_t;
		host_sparse_t m_hostData;

		size_t m_partitionSize;

		size_t m_pitch;

		/* Number of bytes of allocated device memory */
		size_t m_allocated;

		/* Indices of the two planes of the matrix */
		enum {
			RCM_ADDRESS = 0, // source information
			RCM_STDP,
			RCM_FADDRESS,    // actual word address of synapse
			RCM_SUBMATRICES
		};

		/*! \return size (in words) of a single plane of the matrix */
		size_t planeSize() const;

		bool onDevice() const;

		size_t maxSynapsesPerNeuron() const;

		boost::shared_ptr<uint32_t>& allocateDeviceMemory();

		void copyToDevice(
				const std::map<fcm_key_t, class SynapseGroup>& fcm,
				pidx_t targetPartition,
				host_sparse_t h_mem,
				uint32_t* d_mem);
};

} // end namespace nemo

#endif
