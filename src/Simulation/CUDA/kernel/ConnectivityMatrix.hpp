#ifndef CONNECTIVITY_MATRIX_HPP
#define CONNECTIVITY_MATRIX_HPP

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#include "SMatrix.hpp"
#include "RSMatrix.hpp"
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
struct ConnectivityMatrix
{
	public:

		ConnectivityMatrix(
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
				const float* f_weights,
				const uint* f_targetPartition,
				const uint* f_targetNeuron,
				size_t length);

		/* Copy data to device and clear host buffers */
		void moveToDevice();

		/* Copy data from device to host */
		void copyToHost(
				int* f_targetPartition[],
				int* f_targetNeuron[],
				float* f_weights[],
				size_t* pitch);

		/*! \return device data for connectivity */
		uint* df_synapses() const;

		/*! \return device row pitch (in words) */
		size_t df_pitch() const;

		/*! \return the size (in words) for each CM plane (including padding) */
		size_t df_planeSize() const;

		/*! \return device delay bit data */
		uint64_t* df_delayBits() const;

		/*! \return vector specifying maximum synapses per delay (<= pitch) for
		 * each partition */
		const std::vector<uint>& f_maxSynapsesPerDelay() const;

		/*! Clear one plane of connectivity matrix on the device */
		void df_clear(size_t submatrix);
		void clearStdpAccumulator();

		// void printSTDPTrace();

		size_t d_allocated() const;

		/* Per-partition addressing */
		const std::vector<DEVICE_UINT_PTR_T> r_partitionPitch() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionAddress() const;
		const std::vector<DEVICE_UINT_PTR_T> r_partitionStdp() const;

	private:

		SMatrix<uint> m_fsynapses;

		/* We also accumulate the firing delay bits that are used in the spike
		 * delivery */
		NVector<uint64_t> m_delayBits;

		size_t m_partitionCount;
		size_t m_maxPartitionSize;

		unsigned int m_maxDelay;

		/* As we fill the matrix, we accumulate per-partition statistics which
		 * can be used for later configuration */
		std::vector<uint> m_maxSynapsesPerDelay;

		/* For STDP we need a reverse matrix storing source neuron, source
		 * partition, and delay. The reverse connectivity is stored sepearately
		 * for each partition */
		std::vector<RSMatrix> m_rsynapses;

		bool m_setReverse;

		/* The user may want to read back the modified weight matrix. We then
		 * need the corresponding non-compressed addresses as well. The shape
		 * of each of these is exactly that of the weights on the device.
		 * Invalid entries have both partition and neuron set to InvalidNeuron.
		 * */
		std::vector<int> mf_targetPartition;
		std::vector<int> mf_targetNeuron;

		/* The weight matrix is the only bit of data which needs to be read
		 * from the device. This is only allocated if the user requests this
		 * data.  */
		std::vector<uint32_t> mf_weights;

		static const int InvalidNeuron;
};

#endif
