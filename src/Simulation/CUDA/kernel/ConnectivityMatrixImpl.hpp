#ifndef CONNECTIVITY_MATRIX_IMPL_HPP
#define CONNECTIVITY_MATRIX_IMPL_HPP

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#include <map>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <nemo_types.hpp>
#include "nemo_cuda_types.h"
#include "kernel.h" // for synapse type used in interface
#include "SynapseGroup.hpp"
#include "NVector.hpp"
#include "kernel.cu_h"
#include "TargetPartitions.hpp"
#include "L1SpikeBuffer.hpp"


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
class ConnectivityMatrixImpl
{
	public:

		ConnectivityMatrixImpl(
				size_t partitionCount,
				size_t maxPartitionSize,
				bool setReverse);

		/* Set row in both forward and reverse matrix. The input should be
		 * provided in forward order */
		void setRow(
				size_t level,
				uint sourcePartition,
				uint sourceNeuron,
				uint delay,
				const uint* targetPartition,
				const uint* targetNeuron,
				const float* weights,
				const uchar* plastic,
				size_t length);

		delay_t maxDelay() const { return m_maxDelay; }

		/* Copy data to device and clear host buffers */
		void moveToDevice();

		size_t getRow(
				size_t level,
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				uint currentCycle,
				pidx_t* partition[],
				nidx_t* neuron[],
				weight_t* weight[],
				uchar* plastic[]);

		/*! \return device delay bit data */
		uint64_t* df_delayBits(size_t level);

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		/* Per-partition addressing */
		//! \todo no need to return this, set directly, as done in f0_setDispatchTable
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch(size_t lvl) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress(size_t lvl) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp(size_t lvl) const;

		/*! \return pointer to device data containing outgoing spike data for
		 * each neuron */
		targetp_t* outgoing() const { return m_targetp.data(); }

		/*! \return pointer to device data containing the number of outgoing
		 * spike groups for each neuron */
		uint* outgoingCount() const { return m_targetp.count(); }

		/*! \return pointer to device data continaing incoming spike group
		 * buffer for each partition */
		l1spike_t* incoming() const { return m_spikeBuffer.buffer(); }

		/*! \return pointer to device data containing the queue heads (i.e.
		 * the fill) for the incoming spike buffer */
		uint* incomingHeads() const { return m_spikeBuffer.heads(); }

	private:

		/* Add a single synapse to both forward and reverse matrix */
		void addSynapse0(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				weight_t weight,
				uchar isPlastic);

		void addSynapse1(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				weight_t weight,
				uchar isPlastic);

		/* We accumulate the firing delay bits that are used in the spike
		 * delivery */
		NVector<uint64_t> m0_delayBits;
		NVector<uint64_t> m1_delayBits;

		NVector<uint64_t>& delayBits(size_t lvl);

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		delay_t m_maxDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and delay. The reverse connectivity is stored sepearately
		 * for each partition */
		typedef std::vector<class RSMatrix*> rcm_t;
		rcm_t m0_rsynapses;
		rcm_t m1_rsynapses;

		rcm_t& rsynapses(size_t lvl);
		const rcm_t& const_rsynapses(size_t lvl) const;

		bool m_setReverse;

		typedef std::map<nemo::ForwardIdx, SynapseGroup> fcm_t;
		fcm_t m1_fsynapses;

		// new format: smaller groups by source/target/delay
		typedef boost::tuple<pidx_t, pidx_t, delay_t> fcm_key_t; // source, target, delay
		typedef std::map<fcm_key_t, SynapseGroup> fcm1_t;
		fcm1_t m1_fsynapses2;

		void f_setDispatchTable(bool isL0);
		void f1_setDispatchTable();

		boost::shared_ptr<cudaArray> mf1_dispatch;
		boost::shared_ptr<cudaArray> mf1_dispatch2;

		/* For L1 delivery we need to keep track of all target partitions for
		 * each neuron */
		TargetPartitions m_targetp;

		/* We also need device memory for the firing queue */
		L1SpikeBuffer m_spikeBuffer;
};

#endif
