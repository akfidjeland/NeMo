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

#include <boost/format.hpp>

#include <nemo/util.h>
#include <nemo/ConfigurationImpl.hpp>
#include <nemo/fixedpoint.hpp>
#include <nemo/synapse_indices.hpp>

#include "RSMatrix.hpp"
#include "exception.hpp"
#include "WarpAddressTable.hpp"
#include "connectivityMatrix.cu_h"
#include "kernel.hpp"
#include "device_memory.hpp"


namespace nemo {
	namespace cuda {




ConnectivityMatrix::ConnectivityMatrix(
		const nemo::network::NetworkImpl& net,
		const nemo::ConfigurationImpl& conf,
		const Mapper& mapper) :
	m_maxDelay(0),
	mh_weights(WARP_SIZE, 0),
	md_fcmPlaneSize(0),
	md_fcmAllocated(0),
	m_fractionalBits(~0)
{
	//! \todo change synapse_t, perhaps to nidx_dt
	std::vector<synapse_t> h_targets(WARP_SIZE, f_nullSynapse());
	WarpAddressTable wtable;

	bool logging = conf.loggingEnabled();

	m_fractionalBits = conf.fractionalBits();

	if(logging) {
		//! \todo log to correct output stream
		std::cout << "Using fixed point format Q"
			<< 31-m_fractionalBits << "." << m_fractionalBits << " for weights\n";
	}
	CUDA_SAFE_CALL(fx_setFormat(m_fractionalBits));

	/*! \todo perhaps we should reserve a large chunk of memory for
	 * h_targets/h_weights in advance? It's hard to know exactly how much is
	 * needed, though, due the organisation in warp-sized chunks. */

	size_t totalWarps = createFcm(net, mapper, m_fractionalBits, conf.cudaPartitionSize(), wtable, h_targets, mh_weights);

	verifySynapseTerminals(mh_fcmTargets, mapper);

	moveFcmToDevice(totalWarps, h_targets, mh_weights, logging);
	h_targets.clear();

	m_outgoing = Outgoing(mapper.partitionCount(), wtable);
	m_incoming.allocate(mapper.partitionCount(), m_outgoing.maxIncomingWarps(), 1.0);

	moveRcmToDevice();
}



/* Insert into vector, resizing if appropriate */
template<typename T>
void
insert(size_t idx, const T& val, std::vector<T>& vec)
{
	if(idx >= vec.size()) {
		vec.resize(idx+1);
	}
	vec.at(idx) = val;
}




void
ConnectivityMatrix::addSynapse(
		const AxonTerminal& s,
		const DeviceIdx& d_sourceIdx,
		delay_t delay,
		const Mapper& mapper,
		unsigned fbits, //! \todo could remove
		size_t& nextFreeWarp,
		WarpAddressTable& wtable,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights,
		std::vector<unsigned>& h_fcmTargets,
		std::vector<unsigned>& h_fcmDelays,
		std::vector<unsigned char>& h_fcmPlastic,
		std::vector<SynapseAddress>& h_fcmSynapseAddress)
{
	pidx_t d_targetPartition = mapper.deviceIdx(s.target).partition;

	SynapseAddress addr =
		wtable.addSynapse(d_sourceIdx, d_targetPartition, delay, nextFreeWarp);

	if(addr.synapse == 0 && addr.row == nextFreeWarp) {
		nextFreeWarp += 1;
		/* Resize host buffers to accomodate the new warp. This
		 * allocation scheme could potentially result in a
		 * large number of reallocations, so we might be better
		 * off allocating larger chunks here */
		h_targets.resize(nextFreeWarp * WARP_SIZE, f_nullSynapse());
		h_weights.resize(nextFreeWarp * WARP_SIZE, 0);
	}

	//! \todo do more per-synapse computation internally here
	nidx_t d_targetNeuron = mapper.deviceIdx(s.target).neuron;

	size_t f_addr = addr.row * WARP_SIZE + addr.synapse;
	//! \todo range check this address

	h_targets.at(f_addr) = d_targetNeuron;
	h_weights.at(f_addr) = fx_toFix(s.weight, fbits);

	insert(s.id, s.target, h_fcmTargets); // keep global address
	insert(s.id, delay, h_fcmDelays);
	//! \todo store fully computed address here instead
	insert(s.id, addr, h_fcmSynapseAddress);
	insert(s.id, (unsigned char) s.plastic, h_fcmPlastic);

	/*! \todo simplify RCM structure, using a format similar to the FCM */
	//! \todo only need to set this if stdp is enabled
	if(s.plastic) {
		rcm_t& rcm = m_rsynapses;
		if(rcm.find(d_targetPartition) == rcm.end()) {
			rcm[d_targetPartition] = new RSMatrix(mapper.partitionSize());
		}
		rcm[d_targetPartition]->addSynapse(d_sourceIdx, d_targetNeuron, delay, f_addr);
	}
}

size_t
ConnectivityMatrix::createFcm(
		const nemo::network::NetworkImpl& net,
		const Mapper& mapper,
		unsigned fbits,
		size_t partitionSize,
		WarpAddressTable& wtable,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
{
	using boost::format;

	size_t nextFreeWarp = 1; // leave space for null warp at beginning

	for(std::map<nidx_t, network::NetworkImpl::axon_t>::const_iterator axon = net.m_fcm.begin();
			axon != net.m_fcm.end(); ++axon) {

		nidx_t h_sourceIdx = axon->first;
		DeviceIdx d_sourceIdx = mapper.deviceIdx(h_sourceIdx);

		/* Data used when user reads FCM back from device. These are indexed by
		 * (global) synapse ids, and are thus filled in a random order. To
		 * populate these in a single pass over the input, resize on insertion.
		 * The synapse ids are required to form a contigous range, so every
		 * element should be assigned exactly once.
		 */

		//! \todo move access into addSynapse
		//! \todo merge this into a single data structure
		std::vector<unsigned>& h_fcmTargets = mh_fcmTargets[h_sourceIdx];
		std::vector<unsigned>& h_fcmDelays = mh_fcmDelays[h_sourceIdx];
		std::vector<SynapseAddress>& h_fcmSynapseAddress = mh_fcmSynapseAddress[h_sourceIdx];
		std::vector<unsigned char>& h_fcmPlastic = mh_fcmPlastic[h_sourceIdx];

		/* As a simple sanity check, verify that the length of the above data
		 * structures are the same */
		unsigned synapseCount = 0;

		for(std::map<delay_t, network::NetworkImpl::bundle_t>::const_iterator bi = axon->second.begin();
				bi != axon->second.end(); ++bi) {

			delay_t delay = bi->first;

			m_maxDelay = std::max(m_maxDelay, delay);

			if(delay < 1) {
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Neuron %u has synapses with delay < 1 (%u)") % h_sourceIdx % delay));
			}

			network::NetworkImpl::bundle_t bundle = bi->second;

			/* Populate the partition groups. We only need to store the target
			 * neuron and weight. We store these as a pair so that we can
			 * reorganise these later. */
			for(network::NetworkImpl::bundle_t::const_iterator si = bundle.begin();
					si != bundle.end(); ++si) {
				addSynapse(*si, d_sourceIdx, delay, mapper, fbits,
						nextFreeWarp, wtable,
						h_targets, h_weights,
						h_fcmTargets,
						h_fcmDelays,
						h_fcmPlastic,
						h_fcmSynapseAddress);
				synapseCount += 1;
			}
			/*! \todo optionally clear nemo::NetworkImpl data structure as we
			 * iterate over it, so we can work in constant space */
		}

		assert(synapseCount == h_fcmTargets.size());
		assert(synapseCount == h_fcmPlastic.size());
		assert(synapseCount == h_fcmSynapseAddress.size());
		assert(synapseCount == h_fcmDelays.size());
	}

	return nextFreeWarp;
}



void
ConnectivityMatrix::verifySynapseTerminals(
		const std::map<nidx_t, std::vector<nidx_t> >& targets,
		const Mapper& mapper)
{
	using boost::format;

	typedef std::vector<nidx_t> vec;
	typedef std::map<nidx_t, vec>::const_iterator it;
	for(it src_i = targets.begin(); src_i != targets.end(); ++src_i) {

		nidx_t source = src_i->first;
		if(!mapper.valid(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse source neuron %u") % source));
		}

		const vec& row = src_i->second;
		for(vec::const_iterator tgt_i = row.begin(); tgt_i != row.end(); ++tgt_i) {
			nidx_t target = *tgt_i;
			if(!mapper.valid(target)) {
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Invalid synapse target neuron %u (source: %u)") % target % source));
			}
		}
	}
}


void
ConnectivityMatrix::moveFcmToDevice(size_t totalWarps,
		const std::vector<synapse_t>& h_targets,
		const std::vector<weight_dt>& h_weights,
		bool logging)
{
	//! \todo remove warp count from outgoing data structure. It's no longer needed.

	md_fcmPlaneSize = totalWarps * WARP_SIZE;
	size_t bytes = md_fcmPlaneSize * 2 * sizeof(synapse_t);

	synapse_t* d_data;
	d_malloc((void**) &d_data, bytes, "fcm");
	md_fcm = boost::shared_ptr<synapse_t>(d_data, d_free);
	md_fcmAllocated = bytes;

	CUDA_SAFE_CALL(setFcmPlaneSize(md_fcmPlaneSize));

	memcpyToDevice(d_data + md_fcmPlaneSize * FCM_ADDRESS, h_targets, md_fcmPlaneSize);
	memcpyToDevice(d_data + md_fcmPlaneSize * FCM_WEIGHT, h_weights, md_fcmPlaneSize);
}



void
ConnectivityMatrix::moveRcmToDevice()
{
	if(m_rsynapses.size() == 0) {
		return;
	}

	for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
		i->second->moveToDevice();
	}

	CUDA_SAFE_CALL(
		configureReverseAddressing(
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionPitch()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionAddress()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionStdp()[0]),
				const_cast<DEVICE_UINT_PTR_T*>(&r_partitionFAddress()[0]),
				r_partitionPitch().size());
	);
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
		const std::vector<unsigned>**,
		const std::vector<unsigned>**,
		const std::vector<float>**,
		const std::vector<unsigned char>**)
{
	throw nemo::exception(NEMO_API_UNSUPPORTED, "Old-style synapse-query");
}



const std::vector<weight_dt>&
ConnectivityMatrix::syncWeights(cycle_t cycle, const std::vector<synapse_id>& synapses)
{
	if(cycle != m_lastWeightSync && !synapses.empty() && !mh_weights.empty()) {
		//! \todo refine this by only doing the minimal amount of copying
		memcpyFromDevice(&mh_weights[0],
					md_fcm.get() + FCM_WEIGHT * md_fcmPlaneSize,
					md_fcmPlaneSize * sizeof(weight_dt));
		m_lastWeightSync = cycle;
	}
	return mh_weights;
}



const std::vector<float>&
ConnectivityMatrix::getWeights(cycle_t cycle, const std::vector<synapse_id>& synapses)
{
	m_queriedWeights.resize(synapses.size());
	const std::vector<weight_dt>& h_weights = syncWeights(cycle, synapses);
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses.at(i);
		SynapseAddress addr = mh_fcmSynapseAddress[neuronIndex(id)][synapseIndex(id)];
		weight_dt w = h_weights[addr.row * WARP_SIZE + addr.synapse];
		m_queriedWeights[i] = fx_toFloat(w, m_fractionalBits);;
	}
	return m_queriedWeights;
}




template<typename T>
const std::vector<T>&
getSynapseState(
		const std::vector<synapse_id>& synapses,
		const std::map<nidx_t, std::vector<T> >& fcm,
		std::vector<T>& out)
{
	using boost::format;

	out.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses.at(i);
		typename std::map<nidx_t, std::vector<T> >::const_iterator it = fcm.find(neuronIndex(id));
		if(it == fcm.end()) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid neuron id (%u) in synapse query") % neuronIndex(id)));
		}
		out[i] = it->second.at(synapseIndex(id));
	}
	return out;
}



const std::vector<unsigned>&
ConnectivityMatrix::getTargets(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(synapses, mh_fcmTargets, m_queriedTargets);
}


const std::vector<unsigned>&
ConnectivityMatrix::getDelays(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(synapses, mh_fcmDelays, m_queriedDelays);
}


const std::vector<unsigned char>&
ConnectivityMatrix::getPlastic(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(synapses, mh_fcmPlastic, m_queriedPlastic);
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
	if(vec.size() == 0) {
		return std::vector<DEVICE_UINT_PTR_T>();
	}

	pidx_t maxPartitionIdx = vec.rbegin()->first;

	std::vector<DEVICE_UINT_PTR_T> ret(maxPartitionIdx+1, 0);
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
