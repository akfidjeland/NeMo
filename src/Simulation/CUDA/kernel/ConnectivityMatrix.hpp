#ifndef CONNECTIVITY_MATRIX_HPP
#define CONNECTIVITY_MATRIX_HPP

#include <stdint.h>
#include <stddef.h>

#include <vector>

#include "nemo_cuda_types.h"
#include "kernel.cu_h"
#include "outgoing.cu_h"
#include "incoming.cu_h"


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
struct ConnectivityMatrix
{
	public :

		ConnectivityMatrix(
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
				const uint* f_targetPartition,
				const uint* f_targetNeuron,
				const float* f_weights,
				const uchar* f_isPlastic,
				size_t length);

		/* Copy data to device and clear host buffers */
		void moveToDevice();

		/* Copy data from device to host */
		size_t getRow(
				pidx_t sourcePartition,
				nidx_t sourceNeuron,
				delay_t delay,
				uint currentCycle,
				pidx_t* targetPartition[],
				nidx_t* targetNeuron[],
				weight_t* weight[],
				uchar* plastic[]);


		/*! \return device delay bit data */
		uint64_t* df_delayBits(size_t level);

		/*! Clear one plane of connectivity matrix on the device */
		void clearStdpAccumulator();

		size_t d_allocated() const;

		/* Per-partition addressing */
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch(size_t level) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress(size_t level) const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp(size_t level) const;

		delay_t maxDelay() const;

		outgoing_t* outgoing() const;
		uint* outgoingCount() const;
		incoming_t* incoming() const;
		uint* incomingHeads() const;

	private :

		/* We use PIMPL here, so that we can include this header in regular
		 * CUDA code. Internally we use shared_ptr, which causes build errors
		 * for CUDA */
		class ConnectivityMatrixImpl* m_impl;

};

#endif
