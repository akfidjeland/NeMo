#ifndef CONNECTIVITY_MATRIX_IMPL_HPP
#define CONNECTIVITY_MATRIX_IMPL_HPP

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#include <map>
#include <boost/shared_ptr.hpp>

#include <nemo_types.hpp>
#include "nemo_cuda_types.h"
#include "kernel.h" // for synapse type used in interface
#include "SMatrix.hpp"
#include "SynapseGroup.hpp"
#include "NVector.hpp"
#include "kernel.cu_h"

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
				size_t maxDelay,
				size_t maxSynapsesPerDelay,
				bool setReverse);

		/* Set row in both forward and reverse matrix. The input should be
		 * provided in forward order */
		void setRow(
				uint sourcePartition,
				uint sourceNeuron,
				uint delay,
				const uint* targetPartition,
				const uint* targetNeuron,
				const float* weights,
				const uchar* plastic,
				size_t length);

		/* Copy data to device and clear host buffers */
		void moveToDevice(bool isL0);

		size_t getRow(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				uint currentCycle,
				pidx_t* partition[],
				nidx_t* neuron[],
				weight_t* weight[],
				uchar* plastic[]);

		/*! \return device delay bit data */
		uint64_t* df_delayBits() const;

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		/* Per-partition addressing */
		//! \todo no need to return this, set directly, as done in f0_setDispatchTable
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp() const;

	private:

		//! \todo remove this. It's no longer needed
		SMatrix<uint> m_fsynapses;

		/* We also accumulate the firing delay bits that are used in the spike
		 * delivery */
		NVector<uint64_t> m_delayBits;

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		unsigned int m_maxDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and delay. The reverse connectivity is stored sepearately
		 * for each partition */
		std::vector<class RSMatrix*> m_rsynapses;

		bool m_setReverse;

		typedef std::map<nemo::ForwardIdx, SynapseGroup> fcm_t;
		fcm_t m_fsynapses2;

		/* The weight matrix is the only bit of data which needs to be read
		 * from the device. This is only allocated if the user requests this
		 * data.  */
		std::vector<uint32_t> mf_weights;

		void f_setDispatchTable(bool isL0);

		boost::shared_ptr<cudaArray> mf_dispatch;
};

#endif
