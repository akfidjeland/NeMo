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




/* Add a single synapse to both forward and reverse matrix */
void
ConnectivityMatrixImpl::addSynapse(
		size_t level, // 0 or 1
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		pidx_t targetPartition,
		nidx_t targetNeuron,
		weight_t weight,
		uchar isPlastic)
{
	//! \todo make sure caller checks validity of sourcePartition

	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	SynapseGroup& fgroup = fsynapses(level)[nemo::ForwardIdx(sourcePartition, delay)];
	sidx_t sidx = fgroup.addSynapse(sourceNeuron, targetPartition, targetNeuron, weight, isPlastic);

	if(m_setReverse && isPlastic) {
		RSMatrix* rgroup = rsynapses(level)[targetPartition];
		rgroup->addSynapse(sourcePartition, sourceNeuron, sidx, targetNeuron, delay);
	}

	//! \todo factor out delayBits as a separate class
	uint32_t dbits = delayBits(level).getNeuron(sourcePartition, sourceNeuron);
	dbits |= 0x1 << (delay-1);
	delayBits(level).setNeuron(sourcePartition, sourceNeuron, dbits);

	m_maxDelay = std::max(m_maxDelay, delay);

}


void
ConnectivityMatrixImpl::addSynapse0(pidx_t sp, nidx_t sn, delay_t d,
		pidx_t tp, nidx_t tn, weight_t w, uchar plastic)
{
	addSynapse(0, sp, sn, d, tp, tn, w, plastic);
}



void
ConnectivityMatrixImpl::addSynapse1(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		pidx_t targetPartition,
		nidx_t targetNeuron,
		weight_t weight,
		uchar plastic)
{
	addSynapse(1, sourcePartition, sourceNeuron, delay,
			targetPartition, targetNeuron, weight, plastic);

	SynapseGroup& fgroup = m1_fsynapses2[ForwardIdx1(sourcePartition, targetPartition, delay)];

	/* targetPartition not strictly needed here, but left in (in place of
	 * padding) for better code re-use */
	fgroup.addSynapse(sourceNeuron, targetPartition, targetNeuron, weight, plastic);

	m_targetp.addTargetPartition(sourcePartition, sourceNeuron, delay, targetPartition);

	//! \todo add forward index to new reverse matrix
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

	switch(level) {
		case 0 :
			for(size_t i=0; i<f_length; ++i) {
				addSynapse0(sourcePartition, sourceNeuron, delay,
						targetPartition[i], targetNeuron[i], weights[i], isPlastic[i]);
			}
			break;
		case 1 :
			for(size_t i=0; i<f_length; ++i) {
				addSynapse1(sourcePartition, sourceNeuron, delay,
						targetPartition[i], targetNeuron[i], weights[i], isPlastic[i]);
			}
			break;
		default : ERROR("invalid connectivity matrix level");
	}
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

	for(fcm1_t::iterator i = m1_fsynapses2.begin();
			i != m1_fsynapses2.end(); ++i) {
		i->second.moveToDevice();
	}

	//! \todo tidy interface here
	f_setDispatchTable(true);
	f_setDispatchTable(false);
	f1_setDispatchTable();

	m_targetp.moveToDevice(m_partitionCount, m_maxPartitionSize);
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
ConnectivityMatrixImpl::f1_setDispatchTable()
{
	size_t delayCount = MAX_DELAY;

	size_t xdim = m_partitionCount;
	size_t ydim = m_partitionCount;
	size_t zdim = delayCount;
	size_t size = xdim * ydim * zdim;

	fcm_ref_t null = fcm_packReference(0, 0);
	std::vector<fcm_ref_t> table(size, null);

	for(fcm1_t::const_iterator i = m1_fsynapses2.begin(); i != m1_fsynapses2.end(); ++i) {

		ForwardIdx1 fidx = i->first;
		const SynapseGroup& sg = i->second;

		// x: delay, y : target partition, z : source partition
		size_t addr = (fidx.source * m_partitionCount + fidx.target) * delayCount + (fidx.delay-1);

		void* fcm_addr = sg.d_address();
		size_t fcm_pitch = sg.wpitch();
		table.at(addr) = fcm_packReference(fcm_addr, fcm_pitch);
	}

	cudaArray* f_dispatch = ::f1_setDispatchTable2(m_partitionCount, delayCount, table);
	mf1_dispatch2 = boost::shared_ptr<cudaArray>(f_dispatch, cudaFreeArray);
}



void
ConnectivityMatrixImpl::f_setDispatchTable(bool isL0)
{
	size_t delayCount = MAX_DELAY;

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
