/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "WarpAddressTable.hpp"
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
	m_maxPartitionIdx(0),
	m_maxAbsWeight(0.0),
	m_fractionalBits(~0)
{ }




void
ConnectivityMatrix::addSynapse(pidx_t sp, nidx_t sn, delay_t delay,
		pidx_t tp, nidx_t tn, weight_t w, uchar plastic)
{
	//! \todo make sure caller checks validity of sourcePartition
	if(delay > MAX_DELAY || delay == 0) {
		ERROR("delay (%u) out of range (1-%u)", delay, m_maxDelay);
	}

	bundle_t& bundle = mh_fcm[neuron_idx_t(sp, sn)][bundle_idx_t(tp, delay)];
	sidx_t sidx = bundle.size();
	bundle.push_back(synapse_ht(tn, w));
	m_maxAbsWeight = std::max(m_maxAbsWeight, w);
	m_maxPartitionIdx = std::max(m_maxPartitionIdx, std::max(sp, tp));

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
ConnectivityMatrix::addSynapses(
		uint src,
		const std::vector<uint>& targets,
		const std::vector<uint>& delays,
		const std::vector<float>& weights,
		const std::vector<unsigned char> isPlastic)
{
	size_t length = targets.size();
	assert(length == delays.size());
	assert(length == weights.size());
	assert(length == isPlastic.size());

    if(length == 0)
        return;

	pidx_t sourcePartition = partitionIdx(src);

	nidx_t sourceNeuron = neuronIdx(src);
	if(sourceNeuron >= m_maxPartitionSize) {
		ERROR("source neuron index out of range");
	}

	for(size_t i=0; i < length; ++i) {
		pidx_t targetPartition = partitionIdx(targets[i]);
		nidx_t targetNeuron = neuronIdx(targets[i]);
		addSynapse(sourcePartition, sourceNeuron, delays[i],
				targetPartition, targetNeuron, weights[i], isPlastic[i]);
		m_outgoing.addSynapse(sourcePartition, sourceNeuron, delays[i], targetPartition);
	}
}



/* Determine the number of fractional bits to use when storing weights in
 * fixed-point format on the device. */
uint
ConnectivityMatrix::setFractionalBits()
{
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
	uint log2Ceil = ceilf(log2(m_maxAbsWeight));
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
ConnectivityMatrix::moveBundleToDevice(
		const bundle_t& bundle,
		size_t totalWarps,
		uint fbits,
		std::vector<synapse_t>& h_data,
		size_t* woffset)
{
	size_t writtenWarps = 0; // warps

	std::vector<synapse_t> addresses;
	std::vector<weight_dt> weights;

	// fill in addresses and weights in separate vectors
	//! \todo reorganise this for improved memory performance
	for(bundle_t::const_iterator s = bundle.begin(); s != bundle.end(); ++s) {
		addresses.push_back(f_packSynapse(boost::tuples::get<0>(*s)));
		weights.push_back(fixedPoint(boost::tuples::get<1>(*s), fbits));
	}

	assert(sizeof(nidx_t) == sizeof(synapse_t));
	assert(sizeof(weight_dt) == sizeof(synapse_t));

	size_t startWarp = *woffset;
	size_t newWarps = DIV_CEIL(addresses.size(), WARP_SIZE);

	synapse_t* aptr = &h_data.at((startWarp) * WARP_SIZE);
	//! \todo get totalWarps from arg
	synapse_t* wptr = &h_data.at((totalWarps + startWarp) * WARP_SIZE);

	// now copy data into buffer
	/*! note that std::copy won't work as it will silently cast floats to integers */
	memcpy(aptr, &addresses[0], addresses.size() * sizeof(synapse_t));
	memcpy(wptr, &weights[0], weights.size() * sizeof(synapse_t));

	//! \todo also keep track of first warp for this neuron
	//! \todo could write this straight to outgoing?

	*woffset += newWarps;
}




void
ConnectivityMatrix::moveFcmToDevice(WarpAddressTable* warpOffsets)
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

	/* Move all synapses to allocated device data, starting at given warp index.
	 * Return next free warp index */
	//! \todo copy this using a fixed-size buffer (e.g. max 100MB, determine based on PCIx spec)
	size_t woffset1 = 1; // leave space for the null warp
	for(fcm_ht::const_iterator ai = mh_fcm.begin(); ai != mh_fcm.end(); ++ai) {
		pidx_t sourcePartition = boost::tuples::get<0>(ai->first);
		nidx_t sourceNeuron    = boost::tuples::get<1>(ai->first);
		const axon_t& axon = ai->second;
		for(axon_t::const_iterator bundle = axon.begin(); bundle != axon.end(); ++bundle) {
			pidx_t targetPartition = boost::tuples::get<0>(bundle->first);
			delay_t delay          = boost::tuples::get<1>(bundle->first);
			warpOffsets->set(sourcePartition, sourceNeuron, targetPartition, delay, woffset1);
			moveBundleToDevice(bundle->second, totalWarpCount, fbits, h_data, &woffset1);
		}
	}

	md_allocatedFCM = height * bpitch;
	CUDA_SAFE_CALL(cudaMemcpy(d_data, &h_data[0], md_allocatedFCM, cudaMemcpyHostToDevice));

	setFcmPlaneSize(totalWarpCount * wpitch);
	setFixedPointFormat(fbits);
}



void
ConnectivityMatrix::moveToDevice(bool logging)
{
	//! \todo raise exception
	assert(!mh_fcm.empty());

	try {

		// table of initial warp index for different partition/neuron/partition/delay combinations
		//wtable woffsets;
		WarpAddressTable wtable;
		moveFcmToDevice(&wtable);

		for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
			i->second->moveToDevice(wtable, i->first);
		}

		size_t partitionCount = m_maxPartitionIdx + 1;
		size_t maxWarps = m_outgoing.moveToDevice(partitionCount, wtable);

		m_incoming.allocate(partitionCount, maxWarps, 0.1);

		configureReverseAddressing(
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionPitch()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionAddress()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionStdp()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionFAddress()[0]),
				r_partitionPitch().size());

	} catch (DeviceAllocationException& e) {
		std::cerr << e.what() << std::endl;
		printMemoryUsage(std::cerr);
		throw;
	}

	if(logging) {
		//! \todo get output stream from caller
		m_outgoing.reportWarpSizeHistogram(std::cout);
		printMemoryUsage(std::cout);
	}
}



void
ConnectivityMatrix::printMemoryUsage(std::ostream& out) const
{
	const size_t MEGA = 1<<20;
	out << "Memory usage on device:\n";
	out << "forward matrix: " << (md_allocatedFCM / MEGA) << "MB\n";
	out << "reverse matrix: " << (d_allocatedRCM() / MEGA) << "MB (" << m_rsynapses.size() << " groups)\n";
	out << "incoming: " << (m_incoming.allocated() / MEGA) << "MB\n";
	out << "outgoing: " << (m_outgoing.allocated() / MEGA) << "MB\n" << std::endl;
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
