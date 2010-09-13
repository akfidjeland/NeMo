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
#include <nemo/network/Generator.hpp>

#include "types.h"
#include "kernel.cu_h"
#include "Mapper.hpp"
#include "Outgoing.hpp"
#include "Incoming.hpp"
#include "WarpAddressTable.hpp"

namespace nemo {

	class ConfigurationImpl;

	namespace cuda {

		class AxonTerminalAux;

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
				const nemo::network::Generator&,
				const nemo::ConfigurationImpl&,
				const Mapper&);

		delay_t maxDelay() const { return m_maxDelay; }

		/*! \copydoc nemo::Simulation::getTargets */
		const std::vector<unsigned>& getTargets(const std::vector<synapse_id>& synapses);

		/*! \copydoc nemo::Simulation::getDelays */
		const std::vector<unsigned>& getDelays(const std::vector<synapse_id>& synapses);

		/*! \copydoc nemo::Simulation::getWeights */
		const std::vector<float>& getWeights(cycle_t cycle, const std::vector<synapse_id>& synapses);

		/*! \copydoc nemo::Simulation::getPlastic */
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
				const nemo::network::Generator& net,
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

		/* Additional synapse data which is only needed for runtime queries.
		 * Static FCM data for each neuron, required for synapse queries.
		 * Neuron indices are global rather than the partition/neuron scheme
		 * used on the device, so no decoding needs to take place at run-time.
		 */
		typedef std::vector<AxonTerminalAux> aux_row;
		typedef std::map<nidx_t, aux_row> aux_map;
		aux_map m_cmAux;

		void addAuxTerminal(const Synapse&, size_t addr);

#ifndef NDEBUG
		/* Count synapses to verify that m_cmAux contains dense rows */
		std::map<nidx_t, unsigned> m_synapsesPerNeuron;
#endif

		/* Internal buffers for synapse queries */
		std::vector<unsigned> m_queriedTargets;
		std::vector<unsigned> m_queriedDelays;
		std::vector<float> m_queriedWeights;
		std::vector<unsigned char> m_queriedPlastic;

		void addSynapse(
				const Synapse&,
				const Mapper& mapper,
				size_t& nextFreeWarp,
				WarpAddressTable& wtable,
				std::vector<synapse_t>& h_targets,
				std::vector<weight_dt>& h_weights);

		void verifySynapseTerminals(const aux_map&, const Mapper& mapper);
};



/* The parts of the synapse data is only needed if querying synapses at
 * run-time. This data is stored separately */
struct AxonTerminalAux
{
	public :

		unsigned target() const { return m_target; }
		unsigned delay() const { return m_delay; }
		unsigned char plastic() const { return (unsigned char) m_plastic; }
		size_t addr() const { return m_addr; }

		AxonTerminalAux(const Synapse& s, size_t addr) :
			m_target(s.target()), m_delay(s.delay), m_plastic(s.plastic()), m_addr(addr) { }

		AxonTerminalAux() :
			m_target(~0), m_delay(~0), m_plastic(false), m_addr(~0) { }

	private :

		/* Global target index */
		unsigned m_target;

		unsigned m_delay;
		bool m_plastic;

		/* Address into FCM on device */
		size_t m_addr;

};



	} // end namespace cuda
} // end namespace nemo

#endif
