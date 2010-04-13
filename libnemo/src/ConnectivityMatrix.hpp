#ifndef CONNECTIVITY_MATRIX_IMPL_HPP
#define CONNECTIVITY_MATRIX_IMPL_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <nemo_types.hpp>
#include "nemo_cuda_types.h"
#include "NVector.hpp"
#include "kernel.cu_h"
#include "Connectivity.hpp"
#include "Outgoing.hpp"
#include "Incoming.hpp"
#include "SynapseAddressTable.hpp"
#include "WarpAddressTable.hpp"

namespace nemo {
	namespace cuda {

/*! \brief Connectivity matrix
 *
 * The connectivity matrix (CM) specifies how neurons are connected. The CM has
 * both a forward version (connections from presynaptic to postsynaptic) and a
 * reverse version (connetions from postsynaptic to presynaptic). The reverse
 * version may be required if synapses are modified at run time.
 *
 * The CM can have multiple planes of data, e.g. one for addressing and one for
 * synaptic weights.
 *
 * Both the forward and the reverse matrices are stored with synapses organised
 * by:
 *
 * 1. partition
 * 2. neuron
 * 3. delay
 *
 * Functions are prefixed 'f' or 'r' depending on which version it affects.
 *
 * Furthermore, functions are prefixed 'd' or 'h' depending on whether it
 * affects data on the device or on the host.
 */
class ConnectivityMatrix
{
	public:

		ConnectivityMatrix(nemo::Connectivity& cm,
				size_t partitionSize=MAX_PARTITION_SIZE,
				bool logging=false);

		delay_t maxDelay() const { return m_maxDelay; }

		/*! Write all synapse data for a single neuron to output vectors.
		 *
		 * The output vectors are valid until the next call to
		 * this method.
		 *
		 * \post all output vectors have the same length
		 */
		void getSynapses(
				unsigned sourceNeuron, // global index
				const std::vector<unsigned>** targets,
				const std::vector<unsigned>** delays,
				const std::vector<float>** weights,
				const std::vector<unsigned char>** plastic);

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		synapse_t* d_fcm() const { return md_fcm.get(); }

		/*! \return pointer to device data containing outgoing spike data for
		 * each neuron */
		outgoing_t* outgoing() const { return m_outgoing.data(); }

		/*! \return pointer to device data containing the number of outgoing
		 * spike groups for each neuron */
		uint* outgoingCount() const { return m_outgoing.count(); }

		/*! \return pointer to device data continaing incoming spike group
		 * buffer for each partition */
		incoming_t* incoming() const { return m_incoming.buffer(); }

		/*! \return pointer to device data containing the queue heads (i.e.
		 * the fill) for the incoming spike buffer */
		uint* incomingHeads() const { return m_incoming.heads(); }

		/*! \return number of fractional bits used for weights. */
		uint fractionalBits() const { return m_fractionalBits; }

		void printMemoryUsage(std::ostream&) const;

	private:

		delay_t m_maxDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and delay. The reverse connectivity is stored sepearately
		 * for each partition */
		typedef std::map<pidx_t, class RSMatrix*> rcm_t;
		rcm_t m_rsynapses;

		/* Compact fcm on device */
		boost::shared_ptr<synapse_t> md_fcm;
		size_t md_fcmPlaneSize; // in words

		/* For spike delivery we need to keep track of all target partitions
		 * for each neuron */
		Outgoing m_outgoing;

		/* We also need device memory for the firing queue */
		Incoming m_incoming;

		/*! \return Total device memory usage (in bytes) */
		size_t d_allocatedRCM() const;

		size_t md_fcmAllocated;

		uint m_fractionalBits;
		uint setFractionalBits(weight_t wmin, weight_t wmax, bool logging);;

		/* Per-partition addressing of RCM */
		void moveRcmToDevice(const WarpAddressTable& wtable);
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionFAddress() const;

		/* Static FCM data, stored the same order as on the device for each
		 * neuron. This data is used when the user requests synapse data at
		 * run-time. Neuron indices are global rather than the partition/neuron
		 * scheme used on the device, so no decoding needs to take place at
		 * run-time. */
		std::map<nidx_t, std::vector<nidx_t> > mh_fcmTargets;
		std::map<nidx_t, std::vector<delay_t> > mh_fcmDelays;
		std::map<nidx_t, std::vector<uchar> > mh_fcmPlastic;

		/* The weights may change at run-time, so we need to read them back
		 * from the device, and reconstruct possibly disparate data into a
		 * single vector. The synapse addresses table contains the relevant
		 * data for this reconstruction */
		SynapseAddressTable m_synapseAddresses;

		/* We buffer data for only a single source neuron at a time */
		std::vector<weight_t> mh_fcmWeights;

		/* The weight buffer contains the FCM data for a
		 * single neuron, exactly as stored on the device. In
		 * principle, we could load weights for more than one
		 * source neuron at a time, to cut down on PCI traffic
		 * */
		std::vector<synapse_t> mh_weightBuffer;
};

	} // end namespace cuda
} // end namespace nemo

#endif
