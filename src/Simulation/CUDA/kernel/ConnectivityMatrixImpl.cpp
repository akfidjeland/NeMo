#include "ConnectivityMatrixImpl.hpp"

#include "util.h"
#include "log.hpp"
#include "RSMatrix.hpp"
#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "dispatchTable.cu_h"

#include <algorithm>
#include <stdexcept>


ConnectivityMatrixImpl::ConnectivityMatrixImpl(
        size_t partitionCount,
        size_t maxPartitionSize,
		bool setReverse) :
    m0_delayBits(partitionCount, maxPartitionSize, true),
    m1_delayBits(partitionCount, maxPartitionSize, true),
    m_partitionCount(partitionCount),
    m_maxPartitionSize(maxPartitionSize),
    m_maxDelay(0),
	m_setReverse(setReverse)
{
	for(uint p = 0; p < partitionCount; ++p) {
		m0_rsynapses.push_back(new RSMatrix(maxPartitionSize));
		m1_rsynapses.push_back(new RSMatrix(maxPartitionSize));
	}
}



uint64_t*
ConnectivityMatrixImpl::df_delayBits(size_t level)
{
	return delayBits(level).deviceData();
}



NVector<uint64_t>&
ConnectivityMatrixImpl::delayBits(size_t lvl)
{
	switch(lvl) {
		case 0 : return m0_delayBits;
		case 1 : return m1_delayBits;
		default : ERROR("invalid connectivity matrix index");
	}
}



std::vector<class RSMatrix*>&
ConnectivityMatrixImpl::rsynapses(size_t lvl)
{
	switch(lvl) {
		case 0: return m0_rsynapses;
		case 1: return m1_rsynapses;
		default : ERROR("invalid connectivity matrix index");
	}
}



const std::vector<class RSMatrix*>&
ConnectivityMatrixImpl::const_rsynapses(size_t lvl) const
{
	switch(lvl) {
		case 0: return m0_rsynapses;
		case 1: return m1_rsynapses;
		default : ERROR("invalid connectivity matrix index");
	}
}



ConnectivityMatrixImpl::fcm_t&
ConnectivityMatrixImpl::fsynapses(size_t lvl)
{
	switch(lvl) {
		case 0: return m0_fsynapses;
		case 1: return m1_fsynapses;
		default : ERROR("invalid connectivity matrix index");
	}
}



void
ConnectivityMatrixImpl::setRow(
		size_t level, // 0 or 1
		uint sourcePartition,
		uint sourceNeuron,
		uint delay,
		const uint* targetPartition,
		const uint* targetNeuron,
		const float* weights,
		const uchar* isPlastic,
		size_t f_length)
{
	//! \todo do the mapping into l0 and l1 here directly

    if(f_length == 0)
        return;

	if(sourcePartition >= m_partitionCount) {
		ERROR("source partition index out of range");
	}

	if(sourceNeuron >= m_maxPartitionSize) {
		ERROR("source neuron index out of range");
	}

	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	for(size_t i=0; i<f_length; ++i) {
		if(m_setReverse && isPlastic[i]) {
			rsynapses(level)[targetPartition[i]]->addSynapse(
					sourcePartition, sourceNeuron, i,
					targetNeuron[i], delay);
		}
	}

	fsynapses(level)[nemo::ForwardIdx(sourcePartition, delay)].addSynapses(
			sourceNeuron,
			f_length,
			targetPartition,
			targetNeuron,
			weights,
			isPlastic);

	uint32_t dbits = delayBits(level).getNeuron(sourcePartition, sourceNeuron);
	dbits |= 0x1 << (delay-1);
	delayBits(level).setNeuron(sourcePartition, sourceNeuron, dbits);

	m_maxDelay = std::max(m_maxDelay, delay);
}


void
ConnectivityMatrixImpl::moveToDevice()
{
	m0_delayBits.moveToDevice();
	m1_delayBits.moveToDevice();

	for(uint p=0; p < m_partitionCount; ++p){
		m0_rsynapses[p]->moveToDevice();
		m1_rsynapses[p]->moveToDevice();
	}

	//! \todo refactor
	for(fcm_t::iterator i = m0_fsynapses.begin();
			i != m0_fsynapses.end(); ++i) {
		i->second.moveToDevice();
	}

	for(fcm_t::iterator i = m1_fsynapses.begin();
			i != m1_fsynapses.end(); ++i) {
		i->second.moveToDevice();
	}

	//! \todo tidy interface here
	f_setDispatchTable(true);
	f_setDispatchTable(false);
}



size_t
ConnectivityMatrixImpl::getRow(
		size_t lvl,
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		uint currentCycle,
		pidx_t* partition[],
		nidx_t* neuron[],
		weight_t* weight[],
		uchar* plastic[])
{
	fcm_t::iterator group = fsynapses(lvl).find(nemo::ForwardIdx(sourcePartition, delay));
	if(group != fsynapses(lvl).end()) {
		return group->second.getWeights(sourceNeuron, currentCycle, partition, neuron, weight, plastic);
	} else {
		partition = NULL;
		neuron = NULL;
		weight = NULL;
		return 0;
	}
}



void
ConnectivityMatrixImpl::clearStdpAccumulator()
{
	//! \todo this might be done better in a single kernel, to reduce bus traffic
	for(uint p=0; p < m_partitionCount; ++p){
		m0_rsynapses[p]->clearStdpAccumulator();
		m1_rsynapses[p]->clearStdpAccumulator();
	}
}



size_t
ConnectivityMatrixImpl::d_allocated() const
{
	size_t rcm0 = 0;
	for(std::vector<RSMatrix*>::const_iterator i = m0_rsynapses.begin();
			i != m0_rsynapses.end(); ++i) {
		rcm0 += (*i)->d_allocated();
	}

	size_t rcm1 = 0;
	for(std::vector<RSMatrix*>::const_iterator i = m0_rsynapses.begin();
			i != m1_rsynapses.end(); ++i) {
		rcm1 += (*i)->d_allocated();
	}

	size_t fcm0 = 0;
	for(fcm_t::const_iterator i = m0_fsynapses.begin();
			i != m0_fsynapses.end(); ++i) {
		fcm0 += i->second.d_allocated();
	}

	size_t fcm1 = 0;
	for(fcm_t::const_iterator i = m1_fsynapses.begin();
			i != m1_fsynapses.end(); ++i) {
		fcm1 += i->second.d_allocated();
	}

	return m0_delayBits.d_allocated() + m1_delayBits.d_allocated() + fcm0 + fcm1 + rcm0 + rcm1;
}



/* Pack a device pointer to a 32-bit value */
//! \todo replace with non-template version
template<typename T>
DEVICE_UINT_PTR_T
devicePointer(T ptr)
{
	uint64_t ptr64 = (uint64_t) ptr;
#ifndef __DEVICE_EMULATION__
	//! \todo: look up this data at runtime
	//! \todo assert that we can fit all device addresses in 32b address.
	const uint64_t MAX_ADDRESS = 4294967296LL; // on device
	if(ptr64 >= MAX_ADDRESS) {
		throw std::range_error("Device pointer larger than 32 bits");
	}
#endif
	return (DEVICE_UINT_PTR_T) ptr64;

}



/* Map function over vector of reverse synapse matrix */
template<typename T, class S>
const std::vector<DEVICE_UINT_PTR_T>
mapDevicePointer(const std::vector<S*>& vec, std::const_mem_fun_t<T, S> fun)
{
	std::vector<DEVICE_UINT_PTR_T> ret(vec.size(), 0);
	for(uint p=0; p < vec.size(); ++p){
		T ptr = fun(vec[p]);
		ret[p] = devicePointer(ptr);
	}
	return ret;
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionPitch(size_t lvl) const
{
	return mapDevicePointer(const_rsynapses(lvl), std::mem_fun(&RSMatrix::pitch));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionAddress(size_t lvl) const
{
	return mapDevicePointer(const_rsynapses(lvl), std::mem_fun(&RSMatrix::d_address));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrixImpl::r_partitionStdp(size_t lvl) const
{
	return mapDevicePointer(const_rsynapses(lvl), std::mem_fun(&RSMatrix::d_stdp));
}



void
ConnectivityMatrixImpl::f_setDispatchTable(bool isL0)
{
	//! \todo remove magic
	size_t delayCount = 64;

	size_t width = m_partitionCount;
	size_t height = delayCount;
	size_t size = width * height;

	fcm_ref_t null = fcm_packReference(0, 0);
	std::vector<fcm_ref_t> table(size, null);

	size_t lvl = isL0 ? 0 : 1;
	for(fcm_t::const_iterator i = fsynapses(lvl).begin();
			i != fsynapses(lvl).end(); ++i) {

		nemo::ForwardIdx fidx = i->first;
		const SynapseGroup& sg = i->second;

		// x: delay, y : partition
		size_t addr = fidx.source * delayCount + (fidx.delay-1);

		void* fcm_addr = sg.d_address();
		size_t fcm_pitch = sg.wpitch();
		table.at(addr) = fcm_packReference(fcm_addr, fcm_pitch);
	}

	if(isL0) {
		cudaArray* f_dispatch = ::f0_setDispatchTable(m_partitionCount, delayCount, table);
		mf0_dispatch = boost::shared_ptr<cudaArray>(f_dispatch, cudaFreeArray);
	} else {
		cudaArray* f_dispatch = ::f1_setDispatchTable(m_partitionCount, delayCount, table);
		mf1_dispatch = boost::shared_ptr<cudaArray>(f_dispatch, cudaFreeArray);
	}
}
