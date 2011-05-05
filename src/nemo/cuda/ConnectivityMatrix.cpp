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
#include <nemo/cuda/construction/FcmIndex.hpp>

#include "exception.hpp"
#include "connectivityMatrix.cu_h"
#include "kernel.hpp"
#include "device_memory.hpp"
#include "parameters.cu_h"


namespace nemo {
	namespace cuda {


void
setDelays(const construction::FcmIndex& index, NVector<uint64_t>* delays)
{
	using namespace boost::tuples;

	for(construction::FcmIndex::iterator i = index.begin(); i != index.end(); ++i) {
		const construction::FcmIndex::index_key& k = i->first;
		pidx_t p = get<0>(k);
		nidx_t n = get<1>(k);
		delay_t delay1 = get<2>(k);
		uint64_t bits = delays->getNeuron(p, n);
		bits |= (uint64_t(0x1) << uint64_t(delay1-1));
		delays->setNeuron(p, n, bits);
	}
	delays->moveToDevice();
}


ConnectivityMatrix::ConnectivityMatrix(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf,
		const Mapper& mapper) :
	m_mapper(mapper),
	m_maxDelay(0),
	mhf_weights(WARP_SIZE, 0),
	md_fcmPlaneSize(0),
	md_fcmAllocated(0),
	m_delays(1, mapper.partitionCount(), mapper.partitionSize(), true, false),
	m_fractionalBits(conf.fractionalBits()),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	//! \todo change synapse_t, perhaps to nidx_dt
	std::vector<synapse_t> hf_targets(WARP_SIZE, f_nullSynapse());
	construction::FcmIndex fcm_index;

	std::vector<uint32_t> hr_data(WARP_SIZE, INVALID_REVERSE_SYNAPSE);
	std::vector<uint32_t> hr_forward(WARP_SIZE, 0);
	construction::RcmIndex rcm_index;
	size_t r_nextFreeWarp = 1; // leave space for a null warp at the beginning

	bool logging = conf.loggingEnabled();


	if(logging) {
		//! \todo log to correct output stream
		std::cout << "Using fixed point format Q"
			<< 31-m_fractionalBits << "." << m_fractionalBits << " for weights\n";
	}

	/*! \todo perhaps we should reserve a large chunk of memory for
	 * hf_targets/h_weights in advance? It's hard to know exactly how much is
	 * needed, though, due the organisation in warp-sized chunks. */

	size_t nextFreeWarp = 1; // leave space for null warp at beginning
	for(network::synapse_iterator si = net.synapse_begin();
			si != net.synapse_end(); ++si) {
		const Synapse& s = *si;
		setMaxDelay(s);
		DeviceIdx source = mapper.deviceIdx(s.source);
		DeviceIdx target = mapper.deviceIdx(s.target());
		size_t f_addr = addForward(s, source, target, nextFreeWarp, fcm_index, hf_targets, mhf_weights);
		addReverse(s, source, target, f_addr, r_nextFreeWarp, rcm_index,
				hr_data, hr_forward);
		if(!m_writeOnlySynapses) {
			addAuxillary(s, f_addr);
		}
	}

	verifySynapseTerminals(m_cmAux, mapper);

	moveFcmToDevice(nextFreeWarp, hf_targets, mhf_weights);
	hf_targets.clear();

	moveRcmToDevice(r_nextFreeWarp, hr_data, hr_forward);
	hr_data.clear();
	hr_forward.clear();
	m_rcmIndex = runtime::RcmIndex(mapper.partitionCount(), rcm_index);

	md_rcm.data = md_rcmData.get();
	md_rcm.forward = md_rcmForward.get();
	md_rcm.accumulator = md_rcmAccumulator.get();
	md_rcm.index = m_rcmIndex.d_index();
	md_rcm.meta_index = m_rcmIndex.d_metaIndex();

	setDelays(fcm_index, &m_delays);

	m_outgoing = Outgoing(mapper.partitionCount(), fcm_index);
	m_gq.allocate(mapper.partitionCount(), m_outgoing.maxIncomingWarps(), 1.0);

	if(conf.loggingEnabled()) {
		printMemoryUsage(std::cout);
		// fcm_index.reportWarpSizeHistogram(std::cout);
	}
}



void
ConnectivityMatrix::setMaxDelay(const Synapse& s)
{
	using boost::format;

	m_maxDelay = std::max(m_maxDelay, s.delay);

	if(s.delay < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u has synapses with delay < 1 (%u)") % s.source % s.delay));
	}
	if(s.delay > MAX_DELAY) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u has synapses with delay %ums. The CUDA backend supports a maximum of %ums")
						% s.source % s.delay % MAX_DELAY));
	}
}



size_t
ConnectivityMatrix::addForward(
		const Synapse& s,
		const DeviceIdx& d_source,
		const DeviceIdx& d_target,
		size_t& nextFreeWarp,
		construction::FcmIndex& index,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
{
	SynapseAddress addr = index.addSynapse(d_source, d_target.partition, s.delay, nextFreeWarp);

	if(addr.synapse == 0 && addr.row == nextFreeWarp) {
		nextFreeWarp += 1;
		/* Resize host buffers to accomodate the new warp. This
		 * allocation scheme could potentially result in a
		 * large number of reallocations, so we might be better
		 * off allocating larger chunks here */
		h_targets.resize(nextFreeWarp * WARP_SIZE, f_nullSynapse());
		h_weights.resize(nextFreeWarp * WARP_SIZE, 0);
	}

	size_t f_addr = addr.row * WARP_SIZE + addr.synapse;
	//! \todo range check this address

	assert(d_target.neuron < MAX_PARTITION_SIZE);
	h_targets.at(f_addr) = d_target.neuron;
	h_weights.at(f_addr) = fx_toFix(s.weight(), m_fractionalBits);
	return f_addr;
}



void
ConnectivityMatrix::addReverse(
		const Synapse& s,
		const DeviceIdx& d_source,
		const DeviceIdx& d_target,
		size_t f_addr,
		size_t& nextFreeWarp,
		construction::RcmIndex& index,
		std::vector<uint32_t>& sourceData,
		std::vector<uint32_t>& sourceAddress)
{
	//! \todo only need to set this if stdp is enabled
	if(s.plastic()) {

		SynapseAddress addr = index.addSynapse(d_target,  nextFreeWarp);
		if(addr.synapse == 0 && addr.row == nextFreeWarp) {
			nextFreeWarp += 1;
			/* Resize host buffers to accomodate the new warp. This
			 * allocation scheme could potentially result in a
			 * large number of reallocations, so we might be better
			 * off allocating larger chunks here */
			sourceData.resize(nextFreeWarp * WARP_SIZE, INVALID_REVERSE_SYNAPSE);
			sourceAddress.resize(nextFreeWarp * WARP_SIZE, 0);
		}
		//! \todo move this into separate method
		size_t r_addr = addr.row * WARP_SIZE + addr.synapse;
		sourceData.at(r_addr) = r_packSynapse(d_source.partition, d_source.neuron, s.delay);
		sourceAddress.at(r_addr) = f_addr;
	}
}




/*! \note We could verify the synapse terminals during FCM construction. This
 * was found to be somewhat slower, however, as we then end up performing
 * around twice as many checks (since each source is tested many times).
 *
 * If synapses are configured to be write-only, this check will pass, since the
 * CM is empty.
 */
void
ConnectivityMatrix::verifySynapseTerminals(const aux_map& cm, const Mapper& mapper)
{
	using boost::format;

	for(aux_map::const_iterator ni = cm.begin(); ni != cm.end(); ++ni) {

		nidx_t source = ni->first;

		if(!mapper.existingGlobal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse source neuron %u") % source));
		}

		aux_row row = ni->second;

#ifndef NDEBUG
		assert(m_synapsesPerNeuron[source] == row.size());
#endif

		for(aux_row::const_iterator si = row.begin(); si != row.end(); ++si) {
			nidx_t target = si->target();
			if(!mapper.existingGlobal(target)) {
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Invalid synapse target neuron %u (source: %u)") % target % source));
			}
		}
	}
}




void
ConnectivityMatrix::moveFcmToDevice(size_t totalWarps,
		const std::vector<synapse_t>& h_targets,
		const std::vector<weight_dt>& h_weights)
{
	md_fcmPlaneSize = totalWarps * WARP_SIZE;
	size_t bytes = md_fcmPlaneSize * 2 * sizeof(synapse_t);

	void* d_fcm;
	d_malloc(&d_fcm, bytes, "fcm");
	md_fcm = boost::shared_ptr<synapse_t>(static_cast<synapse_t*>(d_fcm), d_free);
	md_fcmAllocated = bytes;

	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_ADDRESS, h_targets, md_fcmPlaneSize);
	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_WEIGHT,
			reinterpret_cast<const synapse_t*>(&h_weights[0]), md_fcmPlaneSize);
}



void
ConnectivityMatrix::moveRcmToDevice(size_t totalWarps,
		const std::vector<uint32_t>& h_data,
		const std::vector<uint32_t>& h_forward)
{
	assert(h_data.size() == h_forward.size());
	assert(totalWarps <= h_data.size() + WARP_SIZE);

	/*! \todo remove the warp counting. Instead just set the plane size based
	 * on the host data. */

	md_rcmPlaneSize = totalWarps * WARP_SIZE;

	md_rcmData = d_array<uint32_t>(md_rcmPlaneSize, "rcm (data)");
	memcpyToDevice(md_rcmData.get(), h_data, md_rcmPlaneSize);

	md_rcmAccumulator = d_array<weight_dt>(md_rcmPlaneSize, "rcm (accumulator)");
	d_memset(md_rcmAccumulator.get(), 0, md_rcmPlaneSize*sizeof(weight_dt));

	md_rcmForward = d_array<uint32_t>(md_rcmPlaneSize, "rcm (forward address)");
	memcpyToDevice(md_rcmForward.get(), h_forward, md_rcmPlaneSize);

	md_rcmAllocated += md_rcmPlaneSize * (2*sizeof(uint32_t) + sizeof(float));
}



void
ConnectivityMatrix::printMemoryUsage(std::ostream& out) const
{
	const size_t MEGA = 1<<20;
	out << "Memory usage on device:\n";
	out << "\tforward matrix: " << (md_fcmAllocated / MEGA) << "MB\n";
	out << "\treverse matrix: " << (d_allocatedRCM() / MEGA) << "MB\n";
	out << "\tglobal queue: " << (m_gq.allocated() / MEGA) << "MB\n";
	out << "\toutgoing: " << (m_outgoing.allocated() / MEGA) << "MB\n" << std::endl;
}



size_t
ConnectivityMatrix::d_allocatedRCM() const
{
	return md_rcmAllocated + m_rcmIndex.d_allocated();
}



/* Data used when user reads FCM back from device. These are indexed by
 * (global) synapse ids, and are thus filled in a random order. To populate
 * these in a single pass over the input, resize on insertion.  The synapse ids
 * are required to form a contigous range, so every element should be assigned
 * exactly once. */
void
ConnectivityMatrix::addAuxillary(const Synapse& s, size_t addr)
{
	id32_t id = s.id();
	aux_row& row= m_cmAux[s.source];
	if(id >= row.size()) {
		row.resize(id+1);
	}
	row.at(id) = AxonTerminalAux(s, addr);
#ifndef NDEBUG
	m_synapsesPerNeuron[s.source] += 1;
#endif
}



const std::vector<synapse_id>&
ConnectivityMatrix::getSynapsesFrom(unsigned source)
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	/* The relevant data is stored in the auxillary synapse map, which is
	 * already indexed in global neuron indices. Therefore, no need to map into
	 * device ids */
	size_t nSynapses = 0;
	aux_map::const_iterator iRow = m_cmAux.find(source);
	if(iRow == m_cmAux.end()) {
		if(!m_mapper.existingGlobal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing source neuron id (%u) in synapse id query") % source));
		}
		/* else just leave nSynapses at zero */
	} else {
		/* Synapse ids are consecutive */
		nSynapses = iRow->second.size();
	}

	m_queriedSynapseIds.resize(nSynapses);

	for(size_t iSynapse = 0; iSynapse < nSynapses; ++iSynapse) {
		m_queriedSynapseIds[iSynapse] = make_synapse_id(source, iSynapse);
	}

	return m_queriedSynapseIds;
}



const std::vector<weight_dt>&
ConnectivityMatrix::syncWeights(cycle_t cycle) const
{
	if(cycle != m_lastWeightSync && !mhf_weights.empty()) {
		//! \todo refine this by only doing the minimal amount of copying
		memcpyFromDevice(reinterpret_cast<synapse_t*>(&mhf_weights[0]),
					md_fcm.get() + FCM_WEIGHT * md_fcmPlaneSize,
					md_fcmPlaneSize);
		m_lastWeightSync = cycle;
	}
	return mhf_weights;
}



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(const synapse_id& id) const
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	aux_map::const_iterator it = m_cmAux.find(neuronIndex(id));
	if(it == m_cmAux.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron id (%u) in synapse query") % neuronIndex(id)));
	}
	return (it->second)[synapseIndex(id)];
}



float
ConnectivityMatrix::getWeight(cycle_t cycle, const synapse_id& id) const
{
	size_t addr = axonTerminalAux(id).addr();
	const std::vector<weight_dt>& h_weights = syncWeights(cycle);
	return fx_toFloat(h_weights[addr], m_fractionalBits);;
}



unsigned
ConnectivityMatrix::getTarget(const synapse_id& id) const
{
	return axonTerminalAux(id).target();
}


unsigned
ConnectivityMatrix::getDelay(const synapse_id& id) const
{
	return axonTerminalAux(id).delay();
}


unsigned char
ConnectivityMatrix::getPlastic(const synapse_id& id) const
{
	return axonTerminalAux(id).plastic();
}


void
ConnectivityMatrix::clearStdpAccumulator()
{
	d_memset(md_rcmAccumulator.get(), 0, md_rcmPlaneSize*sizeof(weight_dt));
}



size_t
ConnectivityMatrix::d_allocated() const
{
	return md_fcmAllocated
		+ d_allocatedRCM()
		+ m_gq.allocated()
		+ m_outgoing.allocated();
}



void
ConnectivityMatrix::setParameters(param_t* params) const
{
	m_outgoing.setParameters(params);
	params->fcmPlaneSize = md_fcmPlaneSize;
	params->rcmPlaneSize = md_rcmPlaneSize;
}


	} // end namespace cuda
} // end namespace nemo
