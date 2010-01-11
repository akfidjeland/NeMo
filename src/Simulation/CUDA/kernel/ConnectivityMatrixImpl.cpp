#include "ConnectivityMatrixImpl.hpp"

#include <algorithm>
#include <stdexcept>
#include "boost/tuple/tuple_comparison.hpp"

#include "util.h"
#include "log.hpp"
#include "RSMatrix.hpp"
#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "dispatchTable.cu_h"


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




boost::tuple<pidx_t, pidx_t, delay_t>
make_fcm_key(pidx_t source, pidx_t target, delay_t delay)
{
	return boost::tuple<pidx_t, pidx_t, delay_t>(source, target, delay);
}


void
ConnectivityMatrixImpl::addSynapse0(pidx_t sp, nidx_t sn, delay_t delay,
		pidx_t tp, nidx_t tn, weight_t w, uchar plastic)
{
	//! \todo make sure caller checks validity of sourcePartition
	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	assert(sp == tp);
	SynapseGroup& fgroup = m1_fsynapses2[make_fcm_key(sp, tp, delay)];

	/* targetPartition not strictly needed here, but left in (in place of
	 * padding) for better code re-use */
	sidx_t sidx = fgroup.addSynapse(sn, tp, tn, w, plastic);

	if(m_setReverse && plastic) {
		RSMatrix* rgroup = rsynapses(0)[tp];
		rgroup->addSynapse(sp, sn, sidx, tn, delay);
	}

	//! \todo factor out delayBits as a separate class
	uint32_t dbits = delayBits(0).getNeuron(sp, sn);
	dbits |= 0x1 << (delay-1);
	delayBits(0).setNeuron(sp, sn, dbits);

	m_maxDelay = std::max(m_maxDelay, delay);
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
	// old format
	//
	//! \todo make sure caller checks validity of sourcePartition
	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	SynapseGroup& fgroup = m1_fsynapses[nemo::ForwardIdx(sourcePartition, delay)];
	sidx_t sidx = fgroup.addSynapse(sourceNeuron, targetPartition, targetNeuron, weight, plastic);

	size_t level = 1;
	if(m_setReverse && plastic) {
		RSMatrix* rgroup = rsynapses(level)[targetPartition];
		rgroup->addSynapse(sourcePartition, sourceNeuron, sidx, targetNeuron, delay);
	}

	//! \todo factor out delayBits as a separate class
	uint32_t dbits = delayBits(level).getNeuron(sourcePartition, sourceNeuron);
	dbits |= 0x1 << (delay-1);
	delayBits(level).setNeuron(sourcePartition, sourceNeuron, dbits);

	m_maxDelay = std::max(m_maxDelay, delay);

	// new format (not yet used)
	{
	SynapseGroup& fgroup = m1_fsynapses2[make_fcm_key(sourcePartition, targetPartition, delay)];

	/* targetPartition not strictly needed here, but left in (in place of
	 * padding) for better code re-use */
	fgroup.addSynapse(sourceNeuron, targetPartition, targetNeuron, weight, plastic);
	}

	m_outgoing.addSynapseGroup(sourcePartition, sourceNeuron, delay, targetPartition);

	//! \todo add forward index to new reverse matrix (see addSynapse0)
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

	for(fcm_t::iterator i = m1_fsynapses.begin();
			i != m1_fsynapses.end(); ++i) {
		i->second.moveToDevice();
	}

	for(fcm1_t::iterator i = m1_fsynapses2.begin();
			i != m1_fsynapses2.end(); ++i) {
		i->second.moveToDevice();
	}

	f1_setDispatchTable();

	m_outgoing.moveToDevice(m_partitionCount);
	m_incoming.allocate(m_partitionCount);
}



size_t
ConnectivityMatrixImpl::getRow(
		pidx_t sourcePartition,
		nidx_t sourceNeuron,
		delay_t delay,
		uint currentCycle,
		pidx_t* partition[],
		nidx_t* neuron[],
		weight_t* weight[],
		uchar* plastic[])
{
	mf_targetPartition.clear();
	mf_targetNeuron.clear();
	mf_weights.clear();
	mf_plastic.clear();

	size_t rowLength = 0;

	// get all synapse groups for which this neuron is present
	for(pidx_t targetPartition = 0; targetPartition < m_partitionCount; ++targetPartition) {
		fcm1_t::iterator group = m1_fsynapses2.find(make_fcm_key(sourcePartition, targetPartition, delay));
		if(group != m1_fsynapses2.end()) {

			pidx_t* pbuf;
			nidx_t* nbuf;
			weight_t* wbuf;
			uchar* sbuf;

			size_t len = group->second.getWeights(sourceNeuron, currentCycle, &pbuf, &nbuf, &wbuf, &sbuf);

			std::copy(pbuf, pbuf+len, back_inserter(mf_targetPartition));
			std::copy(nbuf, nbuf+len, back_inserter(mf_targetNeuron));
			std::copy(wbuf, wbuf+len, back_inserter(mf_weights));
			std::copy(sbuf, sbuf+len, back_inserter(mf_plastic));

			rowLength += len;
		}
	}

	*partition = &mf_targetPartition[0];
	*neuron = &mf_targetNeuron[0];
	*weight = &mf_weights[0];
	*plastic = &mf_plastic[0];
	return rowLength;
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

	size_t fcm = 0;
	for(fcm1_t::const_iterator i = m1_fsynapses2.begin();
			i != m1_fsynapses2.end(); ++i) {
		fcm += i->second.d_allocated();
	}

	return m0_delayBits.d_allocated() + m1_delayBits.d_allocated() + fcm + rcm0 + rcm1;
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

		fcm_key_t fidx = i->first;
		const SynapseGroup& sg = i->second;

		// x: delay, y : target partition, z : source partition
		pidx_t source = boost::tuples::get<0>(fidx);
		pidx_t target = boost::tuples::get<1>(fidx);
		delay_t delay = boost::tuples::get<2>(fidx);
		size_t addr = (source * m_partitionCount + target) * delayCount + (delay-1);

		void* fcm_addr = sg.d_address();
		size_t fcm_pitch = sg.wpitch();
		table.at(addr) = fcm_packReference(fcm_addr, fcm_pitch);
	}

	cudaArray* f_dispatch = ::f1_setDispatchTable2(m_partitionCount, delayCount, table);
	mf1_dispatch2 = boost::shared_ptr<cudaArray>(f_dispatch, cudaFreeArray);
}
