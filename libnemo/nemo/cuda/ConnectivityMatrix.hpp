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

#include <stddef.h>
#include <cuda_runtime.h>

#include <map>
#include <boost/shared_ptr.hpp>

#include <nemo/types.hpp>
#include <nemo/NetworkImpl.hpp>

#include "types.h"
#include "kernel.cu_h"
#include "Mapper.hpp"
#include "Outgoing.hpp"
#include "Incoming.hpp"
#include "WarpAddressTable.hpp"

namespace nemo {

	class ConfigurationImpl;

	namespace cuda {

/*! \brief Connectivity matrix
 *
 * The connectivity matrix (CM) specifies how neurons are connected. The CM has
 * both a forward version (connections from presynaptic to postsynaptic) and a
 * reverse version (connections from postsynaptic to presynaptic). The reverse
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

		ConnectivityMatrix(
				const nemo::network::NetworkImpl&,
				const nemo::ConfigurationImpl&,
				const Mapper&);

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

		const std::vector<unsigned>& getTargets(const std::vector<synapse_id>& synapses);
		const std::vector<unsigned>& getDelays(const std::vector<synapse_id>& synapses);
		const std::vector<float>& getWeights(cycle_t cycle, const std::vector<synapse_id>& synapses);
		const std::vector<unsigned char>& getPlastic(const std::vector<synapse_id>& synapses);

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		synapse_t* d_fcm() const { return md_fcm.get(); }

		/*! \return pointer to device data containing outgoing spike data for
		 * each neuron */
		outgoing_t* outgoing() const { return m_outgoing.data(); }

		/*! \return pointer to device data containing the number of outgoing
		 * spike groups for each neuron */
		unsigned* outgoingCount() const { return m_outgoing.count(); }

		/*! \return pointer to device data continaing incoming spike group
		 * buffer for each partition */
		incoming_t* incoming() const { return m_incoming.buffer(); }

		/*! \return pointer to device data containing the queue heads (i.e.
		 * the fill) for the incoming spike buffer */
		unsigned* incomingHeads() const { return m_incoming.heads(); }

		/*! \return number of fractional bits used for weights. */
		unsigned fractionalBits() const { return m_fractionalBits; }

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

		/* Host-side copy of the weight data. */
		std::vector<weight_dt> mh_weights;

		/* \post The weight of every synapse in 'synapses' is found up-to-date
		 * in mh_weights. */
		const std::vector<weight_dt>& syncWeights(cycle_t, const std::vector<synapse_id>& synapses);
		cycle_t m_lastWeightSync;

		size_t md_fcmPlaneSize; // in words
		size_t md_fcmAllocated; // in bytes

		/*! \return total number of warps */
		size_t createFcm(
				const nemo::network::NetworkImpl& net,
				const Mapper&,
				size_t partitionSize,
				WarpAddressTable& wtable,
				std::vector<synapse_t>& targets,
				std::vector<weight_dt>& weights);

		void moveFcmToDevice(size_t totalWarps,
				const std::vector<synapse_t>& h_targets,
				const std::vector<weight_dt>& h_weights,
				bool logging);

		/* For spike delivery we need to keep track of all target partitions
		 * for each neuron */
		Outgoing m_outgoing;

		/* We also need device memory for the firing queue */
		Incoming m_incoming;

		/*! \return Total device memory usage (in bytes) */
		size_t d_allocatedRCM() const;

		unsigned m_fractionalBits;

		/* Per-partition addressing of RCM */
		void moveRcmToDevice();
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionFAddress() const;

		/* Static FCM data for each neuron, required for synapse queries.
		 * Neuron indices are global rather than the partition/neuron scheme
		 * used on the device, so no decoding needs to take place at run-time.
		 */
		std::map<nidx_t, std::vector<unsigned> > mh_fcmTargets;
		std::map<nidx_t, std::vector<unsigned> > mh_fcmDelays;
		std::map<nidx_t, std::vector<unsigned char> > mh_fcmPlastic;

		/* For the weights we need to look up the data at run-time. We thus
		 * need the warp/synapse index pair */
		std::map<nidx_t, std::vector<SynapseAddress> > mh_fcmSynapseAddress;

		void verifySynapseTerminals(
				const std::map<nidx_t, std::vector<nidx_t> >& targets,
				const Mapper& mapper);

		/* Internal buffers for synapse queries */
		std::vector<unsigned> m_queriedTargets;
		std::vector<unsigned> m_queriedDelays;
		std::vector<float> m_queriedWeights;
		std::vector<unsigned char> m_queriedPlastic;

		void addSynapse(
				const AxonTerminal& s,
				nidx_t source,
				delay_t delay,
				const Mapper& mapper,
				size_t& nextFreeWarp,
				WarpAddressTable& wtable,
				std::vector<synapse_t>& h_targets,
				std::vector<weight_dt>& h_weights);
};



	} // end namespace cuda
} // end namespace nemo

#endif
