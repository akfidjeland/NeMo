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
#include "DeviceIdx.hpp"
#include "WarpAddressTable.hpp"
#include "connectivityMatrix.cu_h"
#include "fixedpoint.hpp"


namespace nemo {
	namespace cuda {



unsigned DeviceIdx::s_partitionSize = MAX_PARTITION_SIZE;


ConnectivityMatrix::ConnectivityMatrix(
		const nemo::Network& net,
		size_t partitionSize,
		bool logging) :
	m_maxDelay(net.maxDelay()),
	md_fcmPlaneSize(0),
	md_fcmAllocated(0),
	m_fractionalBits(~0)
{
	DeviceIdx::setPartitionSize(partitionSize);

	//! \todo change synapse_t, perhaps to nidx_dt
	std::vector<synapse_t> h_targets(WARP_SIZE, f_nullSynapse());
	std::vector<weight_dt> h_weights(WARP_SIZE, 0);
	WarpAddressTable wtable;

	unsigned fbits = setFractionalBits(net.minWeight(), net.maxWeight(), logging);

	/*! \todo perhaps we should reserve a large chunk of memory for
	 * h_targets/h_weights in advance? It's hard to know exactly how much is
	 * needed, though, due the organisation in warp-sized chunks. */

	size_t totalWarps = createFcm(net, fbits, partitionSize, wtable, h_targets, h_weights);

	moveFcmToDevice(totalWarps, h_targets, h_weights, logging);
	h_targets.clear();
	h_weights.clear();

	//! \todo remove need for creating intermediary warp address table. Just
	//construct this directly in m_outgoing.
	//! \todo should we get maxWarps directly in this function?
	size_t partitionCount = DeviceIdx(net.maxSourceIdx()).partition + 1;
	size_t maxWarps = m_outgoing.moveToDevice(partitionCount, wtable);
	m_incoming.allocate(partitionCount, maxWarps, 0.1);

	moveRcmToDevice(wtable);
}




size_t
ConnectivityMatrix::createFcm(
		const nemo::Network& net,
		unsigned fbits,
		size_t partitionSize,
		WarpAddressTable& wtable,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
{
	size_t currentWarp = 1; // leave space for null warp at beginning

	for(std::map<nidx_t, Network::axon_t>::const_iterator axon = net.m_fcm.begin();
			axon != net.m_fcm.end(); ++axon) {

		nidx_t h_sourceIdx = axon->first;
		DeviceIdx d_sourceIdx(h_sourceIdx);
		size_t neuronStartWarp = currentWarp;

		for(std::map<delay_t, Network::bundle_t>::const_iterator bi = axon->second.begin();
				bi != axon->second.end(); ++bi) {

			delay_t delay = bi->first;
			Network::bundle_t bundle = bi->second;

			/* A bundle contains a number of synapses with the same source
			 * neuron and delay. On the device we need to further subdivide
			 * this into groups of synapses with the same target partition */
			std::map<pidx_t, std::vector<synapse_ht> > pgroups;

			/* Populate the partition groups. We only need to store the target
			 * neuron and weight. We store these as a pair so that we can
			 * reorganise these later. */
			for(Network::bundle_t::const_iterator si = bundle.begin();
					si != bundle.end(); ++si) {
				nidx_t h_targetIdx = si->target;
				DeviceIdx d_targetIdx(h_targetIdx);
				weight_dt weight = fx_toFix(si->weight, fbits);
				unsigned char plastic = si->plastic;
				pgroups[d_targetIdx.partition].push_back(synapse_ht(d_targetIdx.neuron, weight, plastic));
			}

			/* Data used when user reads FCM back from device */
			std::vector<nidx_t>& h_fcmTarget = mh_fcmTargets[h_sourceIdx];
			std::vector<unsigned char>& h_fcmPlastic = mh_fcmPlastic[h_sourceIdx];

			for(std::map<pidx_t, std::vector<synapse_ht> >::const_iterator g = pgroups.begin();
					g != pgroups.end(); ++g) {

				pidx_t targetPartition = g->first;
				std::vector<synapse_ht> bundle = g->second;
				size_t warps = DIV_CEIL(bundle.size(), WARP_SIZE);
				size_t words = warps * WARP_SIZE;
				//! \todo change prototype to accept DeviceIdx
				wtable.set(d_sourceIdx.partition, d_sourceIdx.neuron,
						targetPartition, delay, currentWarp);

				//! \todo allocate these only only once (in outer context)
				/* Stage new addresses/weights in temporary buffer. We can re-order
				 * this buffer before writing to h_targets/h_weights in order
				 * to, e.g. optimise for shared memory bank conflicts. */
				std::vector<synapse_t> targets(words, f_nullSynapse());
				std::vector<weight_dt> weights(words, 0);

				for(std::vector<synapse_ht>::const_iterator s = bundle.begin();
						s != bundle.end(); ++s) {
					size_t sidx = s - bundle.begin();
					targets.at(sidx) = s->target;
					weights.at(sidx) = s->weight;
					//! \todo just push the whole targets vector after loop?
					//! \todo remove unused host/device address translation functions
					h_fcmTarget.push_back(DeviceIdx(targetPartition, s->target).hostIdx());
					h_fcmPlastic.push_back(s->plastic);

					//! \todo use a struct for device addresses in outgoing arglist as well
					m_outgoing.addSynapse(d_sourceIdx.partition,
							d_sourceIdx.neuron, delay, targetPartition);

					/*! \todo simplify RCM structure, using a format similar to the FCM */
					//! \todo factor out
					if(s->plastic) {
						rcm_t& rcm = m_rsynapses;
						if(rcm.find(targetPartition) == rcm.end()) {
							rcm[targetPartition] = new RSMatrix(partitionSize);
						}
						//! \todo pass in DeviceIdx here
						rcm[targetPartition]->addSynapse(d_sourceIdx.partition, d_sourceIdx.neuron, sidx, s->target, delay);
					}
				}
				std::fill_n(std::back_inserter(mh_fcmDelays[h_sourceIdx]), bundle.size(), delay);

				/* /Word/ offset relative to first warp for this neuron. In principle this
				 * function could write synapses to a non-contigous range of memory.
				 * However, we currently write this as a single range. */
				size_t bundleStart = (currentWarp - neuronStartWarp) * WARP_SIZE;
				m_synapseAddresses.addBlock(h_sourceIdx, bundleStart, bundleStart + bundle.size());

				std::copy(targets.begin(), targets.end(), back_inserter(h_targets));
				std::copy(weights.begin(), weights.end(), back_inserter(h_weights));

				currentWarp += warps;
			}
			/*! \todo optionally clear nemo::Network data structure as we
			 * iterate over it, so we can work in constant space */
		}

		m_synapseAddresses.setWarpRange(h_sourceIdx, neuronStartWarp, currentWarp);
	}

	return currentWarp;
}



void
ConnectivityMatrix::moveFcmToDevice(size_t totalWarps,
		const std::vector<synapse_t>& h_targets,
		const std::vector<weight_dt>& h_weights,
		bool logging)
{
	//! \todo remove warp count from outgoing data structure. It's no longer needed.
	size_t height = totalWarps * 2; // *2 as we keep target and weight separately
	size_t desiredBytePitch = WARP_SIZE * sizeof(synapse_t);

	// allocate device memory
	size_t bpitch;
	synapse_t* d_data;
	//! \todo wrap this in try/catch block and log memory usage if catching
	cudaError err = cudaMallocPitch((void**) &d_data, &bpitch, desiredBytePitch, height);
	if(cudaSuccess != err) {
		throw DeviceAllocationException("forward connectivity matrix",
				height * desiredBytePitch, err);
	}
	md_fcm = boost::shared_ptr<synapse_t>(d_data, cudaFree);
	size_t wpitch = bpitch / sizeof(synapse_t);
	md_fcmPlaneSize = totalWarps * wpitch;
	setFcmPlaneSize(md_fcmPlaneSize);

	if(logging && bpitch != desiredBytePitch) {
		//! \todo make this an exception.
		//! \todo write this to the correct logging output stream
		std::cout << "Returned byte pitch (" << desiredBytePitch
			<< ") did  not match requested byte pitch (" << bpitch
			<< ") when allocating forward connectivity matrix" << std::endl;
		/* This only matters, as we'll waste memory otherwise, and we'd expect the
		 * desired pitch to always match the returned pitch, since pitch is defined
		 * in terms of warp size */
	}

	md_fcmAllocated = height * bpitch;
	CUDA_SAFE_CALL(cudaMemcpy(d_data + md_fcmPlaneSize * FCM_ADDRESS,
				&h_targets[0], md_fcmPlaneSize*sizeof(synapse_t),
				cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_data + md_fcmPlaneSize * FCM_WEIGHT,
				&h_weights[0], md_fcmPlaneSize*sizeof(synapse_t),
				cudaMemcpyHostToDevice));
}



void
ConnectivityMatrix::moveRcmToDevice(const WarpAddressTable& wtable)
{
	for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
		i->second->moveToDevice(wtable, i->first);
	}

	configureReverseAddressing(
			const_cast<DEVICE_UINT_PTR_T*>(&r_partitionPitch()[0]),
			const_cast<DEVICE_UINT_PTR_T*>(&r_partitionAddress()[0]),
			const_cast<DEVICE_UINT_PTR_T*>(&r_partitionStdp()[0]),
			const_cast<DEVICE_UINT_PTR_T*>(&r_partitionFAddress()[0]),
			r_partitionPitch().size());
}




/* Determine the number of fractional bits to use when storing weights in
 * fixed-point format on the device. */
unsigned
ConnectivityMatrix::setFractionalBits(weight_t wmin, weight_t wmax, bool logging)
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
	weight_t maxAbsWeight = std::max(abs(wmin), abs(wmax));
	unsigned log2Ceil = ceilf(log2(maxAbsWeight));
	unsigned fbits = 31 - log2Ceil - 5; // assumes max 2^5 incoming spikes with max weight

	if(logging) {
		//! \todo log to correct output stream
		std::cout << "Using fixed point format Q"
			<< 31-fbits << "." << fbits << " for weights\n";
	}
	m_fractionalBits = fbits;
	fx_setFormat(fbits);
	return fbits;
}



void
ConnectivityMatrix::printMemoryUsage(std::ostream& out) const
{
	const size_t MEGA = 1<<20;
	out << "Memory usage on device:\n";
	out << "\tforward matrix: " << (md_fcmAllocated / MEGA) << "MB\n";
	out << "\treverse matrix: " << (d_allocatedRCM() / MEGA) << "MB (" << m_rsynapses.size() << " groups)\n";
	out << "\tincoming: " << (m_incoming.allocated() / MEGA) << "MB\n";
	out << "\toutgoing: " << (m_outgoing.allocated() / MEGA) << "MB\n" << std::endl;
}



void
ConnectivityMatrix::getSynapses(
		nidx_t sourceNeuron, // global index
		const std::vector<unsigned>** targets,
		const std::vector<unsigned>** delays,
		const std::vector<float>** weights,
		const std::vector<unsigned char>** plastic)
{
	//! \todo assert that we have moved onto device
	AddressRange warps = m_synapseAddresses.warpsOf(sourceNeuron);
	size_t words = warps.size() * WARP_SIZE;

	mh_weightBuffer.resize(words);
	CUDA_SAFE_CALL(cudaMemcpy(&mh_weightBuffer[0],
				md_fcm.get() + FCM_WEIGHT * md_fcmPlaneSize + warps.start * WARP_SIZE,
				words * sizeof(synapse_t),
				cudaMemcpyDeviceToHost));
	/*! \todo read back data for more than one neuron. Keep
	 * track of what cycle we last accessed each neuron and
	 * what device data is currently cached here. */

	// fill in weights
	assert(m_fractionalBits != ~0U);
	mh_fcmWeights.clear();
	const std::vector<AddressRange>& ranges = m_synapseAddresses.synapsesOf(sourceNeuron);
	for(std::vector<AddressRange>::const_iterator i = ranges.begin();
			i != ranges.end(); ++i) {
		for(unsigned addr = i->start; addr < i->end; ++addr) {
			weight_t w = fx_toFloat(mh_weightBuffer[addr], m_fractionalBits);
			mh_fcmWeights.push_back(w);
		}
	}

	*targets = &mh_fcmTargets[sourceNeuron];
	*delays = &mh_fcmDelays[sourceNeuron];
	*weights = &mh_fcmWeights;
	*plastic = &mh_fcmPlastic[sourceNeuron];
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
	return md_fcmAllocated
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


	} // end namespace cuda
} // end namespace nemo
