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
#include "NVector.hpp"
#include "kernel.cu_h"
#include "Outgoing.hpp"
#include "Incoming.hpp"


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

	private:

		/* Per-partition addressing */
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch(size_t lvl) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress(size_t lvl) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp(size_t lvl) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionFAddress(size_t lvl) const;

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

	private:

		/* Add a single synapse to both forward and reverse matrix */
		void addSynapse(
				size_t lvl,
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				pidx_t targetPartition,
				nidx_t targetNeuron,
				weight_t weight,
				uchar isPlastic);

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

		/* The connectivity matrix is stored in small blocks specific to a
		 * combination of source partiton, target partition and delay */
		typedef boost::tuple<pidx_t, pidx_t, delay_t> fcm_key_t; // source, target, delay
		typedef std::map<fcm_key_t, class SynapseGroup> fcm_t;
		fcm_t m_fsynapses;

		/* Compact fcm on device */
		boost::shared_ptr<synapse_t> md_fcm;

		void f_setDispatchTable();

		boost::shared_ptr<cudaArray> mf_dispatch;

		/* For L1 delivery we need to keep track of all target partitions for
		 * each neuron */
		Outgoing m_outgoing;

		/* We also need device memory for the firing queue */
		Incoming m_incoming;

		/* When the user requests a row of synapses we need to combine data
		 * from several synapse groups */
		std::vector<pidx_t> mf_targetPartition;
		std::vector<nidx_t> mf_targetNeuron;
		std::vector<uchar> mf_plastic;
		std::vector<weight_t> mf_weights;

		/* Memory usage. All values in bytes */
		size_t d_allocatedFCM() const;
		size_t d_allocatedRCM0() const;
		size_t d_allocatedRCM1() const;
		void printMemoryUsage(FILE* out);

		void moveFcmToDevice();

		size_t md_allocatedFCM;
};

#endif
