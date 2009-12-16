#include "ConnectivityMatrix.hpp"
#include "ConnectivityMatrixImpl.hpp"

ConnectivityMatrix::ConnectivityMatrix(
        size_t partitionCount,
        size_t maxPartitionSize,
		size_t maxDelay,
		size_t maxSynapsesPerDelay,
		bool setReverse) :
	m_impl(new ConnectivityMatrixImpl(partitionCount,
		maxPartitionSize, maxDelay, maxSynapsesPerDelay,
		setReverse)) {}


void
ConnectivityMatrix::setRow(
		uint sourcePartition,
		uint sourceNeuron,
		uint delay,
		const float* f_weights,
		const uint* f_targetPartition,
		const uint* f_targetNeuron,
		const uint* f_isPlastic,
		size_t length)
{
	m_impl->setRow(sourcePartition,
		sourceNeuron,
		delay,
		f_weights,
		f_targetPartition,
		f_targetNeuron,
		f_isPlastic,
		length);
}


void
ConnectivityMatrix::moveToDevice(bool isL0)
{
	m_impl->moveToDevice(isL0);
}


void
ConnectivityMatrix::copyToHost(
		int* f_targetPartition[],
		int* f_targetNeuron[],
		float* f_weights[],
		size_t* pitch)
{
	m_impl->copyToHost(f_targetPartition, f_targetNeuron, f_weights, pitch);
}



uint64_t*
ConnectivityMatrix::df_delayBits() const
{
	return m_impl->df_delayBits();
}


void
ConnectivityMatrix::df_clear(size_t submatrix)
{
	return m_impl->df_clear(submatrix);
}


void
ConnectivityMatrix::clearStdpAccumulator()
{
	m_impl->clearStdpAccumulator();
}


size_t
ConnectivityMatrix::d_allocated() const
{
	return m_impl->d_allocated();
}


const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionPitch() const
{
	return m_impl->r_partitionPitch();
}


const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionAddress() const
{
	return m_impl->r_partitionAddress();
}


const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionStdp() const
{
	return m_impl->r_partitionStdp();
}
