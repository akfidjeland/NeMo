#include "ConnectivityMatrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <boost/tuple/tuple_comparison.hpp>

#include "util.h"
#include "log.hpp"
#include "RSMatrix.hpp"
#include "except.hpp"
#include "SynapseGroup.hpp"
#include "connectivityMatrix.cu_h"
#include "fixedpoint.hpp"

namespace nemo {

ConnectivityMatrix::ConnectivityMatrix(
        size_t maxPartitionSize,
		bool setReverse) :
    m_maxPartitionSize(maxPartitionSize),
    m_maxDelay(0),
	m_setReverse(setReverse),
	md_allocatedFCM(0),
	m_fractionalBits(~0)
{ }



boost::tuple<pidx_t, pidx_t, delay_t>
make_fcm_key(pidx_t source, pidx_t target, delay_t delay)
{
	return boost::tuple<pidx_t, pidx_t, delay_t>(source, target, delay);
}


void
ConnectivityMatrix::addSynapse(pidx_t sp, nidx_t sn, delay_t delay,
		pidx_t tp, nidx_t tn, weight_t w, uchar plastic)
{
	//! \todo make sure caller checks validity of sourcePartition
	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	SynapseGroup& fgroup = m_fsynapses[make_fcm_key(sp, tp, delay)];

	/* targetPartition not strictly needed here, but left in (in place of
	 * padding) for better code re-use */
	sidx_t sidx = fgroup.addSynapse(sn, tp, tn, w, plastic);

	if(m_setReverse && plastic) {
		/*! \todo should modify RSMatrix so that we don't need the partition
		 * size until we move to device */
		//! \todo simplify
		rcm_t& rcm = m_rsynapses;
		if(rcm.find(tp) == rcm.end()) {
			rcm[tp] = new RSMatrix(m_maxPartitionSize);
		}
		rcm[tp]->addSynapse(sp, sn, sidx, tn, delay);
	}

	m_maxDelay = std::max(m_maxDelay, delay);
}



void
ConnectivityMatrix::setRow(
		uint src,
		const uint* tgt,
		const uint* delay,
		const float* weights,
		const uchar* isPlastic,
		size_t f_length)
{
	//! \todo do the mapping into l0 and l1 here directly

    if(f_length == 0)
        return;

	pidx_t sourcePartition = partitionIdx(src);

	nidx_t sourceNeuron = neuronIdx(src);
	if(sourceNeuron >= m_maxPartitionSize) {
		ERROR("source neuron index out of range");
	}

	for(size_t i=0; i<f_length; ++i) {
		pidx_t targetPartition = partitionIdx(tgt[i]);
		nidx_t targetNeuron = neuronIdx(tgt[i]);
		addSynapse(sourcePartition, sourceNeuron, delay[i],
				targetPartition, targetNeuron, weights[i], isPlastic[i]);
		m_outgoing.addSynapse(sourcePartition, sourceNeuron, delay[i], targetPartition);
	}
}



/* Determine the number of fractional bits to use when storing weights in
 * fixed-point format on the device. */
uint
ConnectivityMatrix::setFractionalBits()
{
	weight_t maxAbsWeight = 0.0f;
	for(fcm_t::const_iterator i = m_fsynapses.begin(); i != m_fsynapses.end(); ++i) {
		maxAbsWeight = std::max(maxAbsWeight, i->second.maxAbsWeight());
	}

	/* In the worst case we may have all presynaptic neurons for some neuron
	 * firing, and having all the relevant synapses have the maximum weight we
	 * just computed. Based on this, it's possible to set the radix point such
	 * that we are guaranteed never to overflow. However, if we optimise for
	 * this pathological case we'll end up throwing away precision for no
	 * appreciable gain. Instead we rely on overflow detection on the device
	 * (which will lead to saturation of the input current).
	 *
	 * We can make some reasonable assumptions regarding the number of neurons
	 * expected to fire at any time as well as the distribution of weights.
	 *
	 * For now just assume that at most a fixed number of neurons will fire at
	 * max weight. */
	//! \todo do this based on both max weight and max number of incoming synapses
	uint log2Ceil = ceilf(log2(maxAbsWeight));
	uint fbits = 31 - log2Ceil - 5; // assumes max 2^5 incoming spikes with max weight
	//! \todo log this to file
	//fprintf(stderr, "Using fixed point format %u.%u\n", 31-fbits, fbits);
	m_fractionalBits = fbits;
	return fbits;
}



uint
ConnectivityMatrix::fractionalBits() const
{
	if(m_fractionalBits == ~0U) {
		throw std::runtime_error("Fractional bits requested before it was set");
	}
	return m_fractionalBits;
}


void
ConnectivityMatrix::moveFcmToDevice()
{
	/* We add 1 extra warp here, so we can leave a null warp at the beginning */
	size_t totalWarpCount = 1 + m_outgoing.totalWarpCount();

	size_t height = totalWarpCount * 2; // *2 as we keep address and weights separately
	size_t desiredBytePitch = WARP_SIZE * sizeof(synapse_t);

	size_t bpitch;
	synapse_t* d_data;

	// allocate device memory
	cudaError err = cudaMallocPitch((void**) &d_data,
				&bpitch,
				desiredBytePitch,
				height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("forward connectivity matrix",
				height * desiredBytePitch, err);
	}
	md_fcm = boost::shared_ptr<synapse_t>(d_data, cudaFree);

	if(bpitch != desiredBytePitch) {
		std::cerr << "Returned byte pitch (" << desiredBytePitch
			<< ") did  not match requested byte pitch (" << bpitch
			<< ") when allocating forward connectivity matrix" << std::endl;
		/* This only matters, as we'll waste memory otherwise, and we'd expect the
		 * desired pitch to always match the returned pitch, since pitch is defined
		 * in terms of warp size */
	}

	// allocate and intialise host memory
	size_t wpitch = bpitch / sizeof(synapse_t);
	std::vector<synapse_t> h_data(height * wpitch, f_nullSynapse());

	uint fbits = setFractionalBits();

	size_t woffset = 1; // leave space for the null warp
	for(fcm_t::iterator i = m_fsynapses.begin(); i != m_fsynapses.end(); ++i) {
		woffset += i->second.fillFcm(fbits, woffset, totalWarpCount, h_data);
	}

	md_allocatedFCM = height * bpitch;
	CUDA_SAFE_CALL(cudaMemcpy(d_data, &h_data[0], md_allocatedFCM, cudaMemcpyHostToDevice));

	setFcmPlaneSize(totalWarpCount * wpitch);
	setFixedPointFormat(fbits);
}



pidx_t
ConnectivityMatrix::maxPartitionIdx() const
{
	pidx_t maxIdx = 0;
	for(fcm_t::const_iterator i = m_fsynapses.begin();
			i != m_fsynapses.end(); ++i) {
		maxIdx = std::max(maxIdx, boost::get<0>(i->first));
		maxIdx = std::max(maxIdx, boost::get<1>(i->first));
	}
	return maxIdx;
}


void
ConnectivityMatrix::moveToDevice()
{
	try {
		moveFcmToDevice();

		for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
			i->second->moveToDevice(m_fsynapses, i->first);
		}

		size_t partitionCount = maxPartitionIdx() + 1;
		size_t maxWarps = m_outgoing.moveToDevice(partitionCount, m_fsynapses);
		m_incoming.allocate(partitionCount, maxWarps, 0.1);

		configureReverseAddressing(
				r_partitionPitch(),
				r_partitionAddress(),
				r_partitionStdp(),
				r_partitionFAddress());

	} catch (DeviceAllocationException& e) {
		FILE* out = stderr;
		fprintf(out, e.what());
		printMemoryUsage(out);
		throw;
	}
	//! \todo remove debugging code
	//printMemoryUsage(stderr);
}



void
ConnectivityMatrix::printMemoryUsage(FILE* out)
{
	const size_t MEGA = 1<<20;
	fprintf(out, "forward matrix:     %6luMB\n",
			md_allocatedFCM / MEGA);
	fprintf(out, "reverse matrix:     %6luMB (%lu groups)\n",
			d_allocatedRCM() / MEGA, m_rsynapses.size());
	fprintf(out, "incoming:           %6luMB\n", m_incoming.allocated() / MEGA);
	fprintf(out, "outgoing:           %6luMB\n", m_outgoing.allocated() / MEGA);
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
	//! \todo need to add this back!
#if 0
	mf_targetPartition.clear();
	mf_targetNeuron.clear();
	mf_weights.clear();
	mf_plastic.clear();

	size_t rowLength = 0;

	// get all synapse groups for which this neuron is present
	for(pidx_t targetPartition = 0; targetPartition < m_partitionCount; ++targetPartition) {
		fcm_t::iterator group = m_fsynapses.find(make_fcm_key(sourcePartition, targetPartition, delay));
		if(group != m_fsynapses.end()) {

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
#endif
	return 0;
}



void
ConnectivityMatrix::clearStdpAccumulator()
{
	for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
		i->second->clearStdpAccumulator();
	}
#if 0
	for(rcm_t::const_iterator i = m0_rsynapses.begin(); i != m0_rsynapses.end(); ++i) {
		i->second->clearStdpAccumulator();
	}
	for(rcm_t::const_iterator i = m1_rsynapses.begin(); i != m1_rsynapses.end(); ++i) {
		i->second->clearStdpAccumulator();
	}
#endif
}


size_t
ConnectivityMatrix::d_allocatedRCM() const
{
	size_t bytes = 0;
	for(std::map<pidx_t, RSMatrix*>::const_iterator i = m_rsynapses.begin();
			i != m_rsynapses.end(); ++i) {
		bytes += i->second->d_allocated();
	}
	return bytes;
}



size_t
ConnectivityMatrix::d_allocated() const
{

	return md_allocatedFCM
		+ d_allocatedRCM()
		+ m_incoming.allocated()
		+ m_outgoing.allocated();
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
mapDevicePointer(const std::map<pidx_t, S*>& vec, std::const_mem_fun_t<T, S> fun)
{
	std::vector<DEVICE_UINT_PTR_T> ret(vec.size(), 0);
	for(typename std::map<pidx_t, S*>::const_iterator i = vec.begin();
			i != vec.end(); ++i) {
		T ptr = fun(i->second);
		ret.at(i->first) = devicePointer(ptr);
	}
	return ret;
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionPitch() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::pitch));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionAddress() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_address));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionStdp() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_stdp));
}



const std::vector<DEVICE_UINT_PTR_T>
ConnectivityMatrix::r_partitionFAddress() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_faddress));
}



nidx_t
ConnectivityMatrix::neuronIdx(nidx_t nidx)
{
	return nidx % m_maxPartitionSize;	
}



pidx_t
ConnectivityMatrix::partitionIdx(pidx_t pidx)
{
	return pidx / m_maxPartitionSize;	
}

} // end namespace nemo
