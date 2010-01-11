#include "ConnectivityMatrix.hpp"
#include "ConnectivityMatrixImpl.hpp"


ConnectivityMatrix::ConnectivityMatrix(
        size_t partitionCount,
        size_t maxPartitionSize,
		bool setReverse) :
	m_impl(new ConnectivityMatrixImpl(partitionCount,
		maxPartitionSize, setReverse)) {}


void
ConnectivityMatrix::setRow(
		size_t level,
		uint sourcePartition,
		uint sourceNeuron,
		uint delay,
		const uint* f_targetPartition,
		const uint* f_targetNeuron,
		const float* f_weights,
		const uchar* f_isPlastic,
		size_t length)
{
	m_impl->setRow(level,
		sourcePartition,
		sourceNeuron,
		delay,
		f_targetPartition,
		f_targetNeuron,
		f_weights,
		f_isPlastic,
		length);
}



void
ConnectivityMatrix::moveToDevice()
{
	m_impl->moveToDevice();
}



size_t
ConnectivityMatrix::getRow(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		uint currentCycle,
		pidx_t* partition[],
		nidx_t* neuron[],
		weight_t* weight[],
		uchar* plastic[])
{
	return m_impl->getRow(sourcePartition, sourceNeuron,
			delay, currentCycle, partition, neuron, weight, plastic);
}



uint64_t*
ConnectivityMatrix::df_delayBits(size_t level)
{
	return m_impl->df_delayBits(level);
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
ConnectivityMatrix::r_partitionPitch(size_t level) const
{
	return m_impl->r_partitionPitch(level);
}


const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionAddress(size_t level) const
{
	return m_impl->r_partitionAddress(level);
}


const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionStdp(size_t level) const
{
	return m_impl->r_partitionStdp(level);
}


delay_t
ConnectivityMatrix::maxDelay() const
{
	return m_impl->maxDelay();
}



outgoing_t*
ConnectivityMatrix::outgoing() const
{
	return m_impl->outgoing();
}



uint*
ConnectivityMatrix::outgoingCount() const
{
	return m_impl->outgoingCount();
}



incoming_t*
ConnectivityMatrix::incoming() const
{
	return m_impl->incoming();
}



uint*
ConnectivityMatrix::incomingHeads() const
{
	return m_impl->incomingHeads();
}
