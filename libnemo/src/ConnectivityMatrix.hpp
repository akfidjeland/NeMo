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
#include "Outgoing.hpp"
#include "Incoming.hpp"
#include "SynapseAddressTable.hpp"

namespace nemo {

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

		ConnectivityMatrix(size_t maxPartitionSize, bool setReverse);

		/* Set row in both forward and reverse matrix. The input should be
		 * provided in forward order */
		void addSynapses(
				uint sourceNeuron, // global neuron indices
				const std::vector<uint>& targets,
				const std::vector<uint>& delays,
				const std::vector<float>& weights,
				const std::vector<unsigned char> is_plastic);

		delay_t maxDelay() const { return m_maxDelay; }

		/*! Copy data to device and clear host buffers. If \a
		 * logging is enabled, print info (memory usage etc.)
		 * to stdout */
		void moveToDevice(bool logging);

		size_t getRow(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				uint currentCycle,
				pidx_t* partition[],
				nidx_t* neuron[],
				weight_t* weight[],
				uchar* plastic[]);

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

	public:

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

		/*! \return number of fractional bits used for weights. This is only
		 * known once the FCM is finalised, i.e. once moveToDevice has been
		 * called */
		uint fractionalBits() const;

		void printMemoryUsage(std::ostream&) const;

	private:

		/* Add a single synapse to both forward and reverse matrix */
		void addSynapse(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				weight_t weight,
				uchar isPlastic);

		size_t m_maxPartitionSize;

		delay_t m_maxDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and delay. The reverse connectivity is stored sepearately
		 * for each partition */
		typedef std::map<pidx_t, class RSMatrix*> rcm_t;
		rcm_t m_rsynapses;

		bool m_setReverse;

	private:

		typedef boost::tuple<nidx_t, weight_t, uchar> synapse_ht;

		typedef std::vector<synapse_ht> bundle_t;
		typedef boost::tuple<pidx_t, delay_t> bundle_idx_t; // target partition, target delay

		typedef std::map<bundle_idx_t, bundle_t> axon_t;
		typedef boost::tuple<pidx_t, nidx_t> neuron_idx_t; // source partition, source neuron

		typedef std::map<neuron_idx_t, axon_t> fcm_ht;
		fcm_ht mh_fcm;

		/* Compact fcm on device */
		boost::shared_ptr<synapse_t> md_fcm;

		/* For spike delivery we need to keep track of all target partitions
		 * for each neuron */
		Outgoing m_outgoing;

		/* We also need device memory for the firing queue */
		Incoming m_incoming;

		/*! \return Total device memory usage (in bytes) */
		size_t d_allocatedRCM() const;

		void moveFcmToDevice(class WarpAddressTable*);
		void moveBundleToDevice(
				nidx_t globalSourceNeuron,
				pidx_t targetPartition,
				delay_t delay,
				const bundle_t& bundle,
				size_t totalWarps,
				size_t axonStart,
				uint fbits,
				std::vector<synapse_t>& h_data,
				size_t* woffset);

		size_t md_allocatedFCM;

		/* Convert global neuron index to local neuron index */
		nidx_t neuronIdx(nidx_t);

		/* Convert global neuron index to partition index */
		pidx_t partitionIdx(nidx_t);
		pidx_t m_maxPartitionIdx;

		nidx_t globalIndex(pidx_t p, nidx_t n);

		weight_t m_maxAbsWeight;
		uint m_fractionalBits;
		uint setFractionalBits();

		/* Per-partition addressing of RCM */
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
		std::map<nidx_t, std::vector<uchar> > mh_fcmStatic;

		/* The weights may change at run-time, so we need to read them back
		 * from the device, and reconstruct possibly disparate data into a
		 * single vector. The synapse addresses contains the relevant data for
		 * this reconstruction. */
		SynapseAddressTable m_synapseAddresses;

};

} // end namespace nemo

#endif
