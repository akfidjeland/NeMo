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


void
setDelays(const WarpAddressTable& wtable, NVector<uint64_t>* delays)
{
	using namespace boost::tuples;

	for(WarpAddressTable::iterator i = wtable.begin(); i != wtable.end(); ++i) {
		const WarpAddressTable::key& k = i->first;
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
	mh_weights(WARP_SIZE, 0),
	md_fcmPlaneSize(0),
	md_fcmAllocated(0),
	m_delays(mapper.partitionCount(), mapper.partitionSize(), true, false),
	m_fractionalBits(conf.fractionalBits()),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	//! \todo change synapse_t, perhaps to nidx_dt
	std::vector<synapse_t> h_targets(WARP_SIZE, f_nullSynapse());
	WarpAddressTable wtable;

	bool logging = conf.loggingEnabled();


	if(logging) {
		//! \todo log to correct output stream
		std::cout << "Using fixed point format Q"
			<< 31-m_fractionalBits << "." << m_fractionalBits << " for weights\n";
	}
	CUDA_SAFE_CALL(fx_setFormat(m_fractionalBits));

	/*! \todo perhaps we should reserve a large chunk of memory for
	 * h_targets/h_weights in advance? It's hard to know exactly how much is
	 * needed, though, due the organisation in warp-sized chunks. */

	size_t totalWarps = createFcm(net, mapper, wtable, h_targets, mh_weights);

	verifySynapseTerminals(m_cmAux, mapper);

	moveFcmToDevice(totalWarps, h_targets, mh_weights, logging);
	h_targets.clear();

	setDelays(wtable, &m_delays);

	m_outgoing = Outgoing(mapper.partitionCount(), wtable);
	m_gq.allocate(mapper.partitionCount(), m_outgoing.maxIncomingWarps(), 1.0);

	moveRcmToDevice();

	if(conf.loggingEnabled()) {
		printMemoryUsage(std::cout);
		// wtable.reportWarpSizeHistogram(std::cout);
	}
}



ConnectivityMatrix::~ConnectivityMatrix()
{
	for(rcm_t::iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
		delete(i->second);
	}
}



void
ConnectivityMatrix::addSynapse(
		const Synapse& s,
		const Mapper& mapper,
		size_t& nextFreeWarp,
		WarpAddressTable& wtable,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
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

	DeviceIdx d_target = mapper.deviceIdx(s.target());
	DeviceIdx d_source = mapper.deviceIdx(s.source);

	SynapseAddress addr = wtable.addSynapse(d_source, d_target.partition, s.delay, nextFreeWarp);

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

	if(!m_writeOnlySynapses) {
		addAuxTerminal(s, f_addr);
	}

	/*! \todo simplify RCM structure, using a format similar to the FCM */
	//! \todo only need to set this if stdp is enabled
	if(s.plastic()) {
		rcm_t& rcm = m_rsynapses;
		if(rcm.find(d_target.partition) == rcm.end()) {
			rcm[d_target.partition] = new RSMatrix(mapper.partitionSize());
		}
		rcm[d_target.partition]->addSynapse(d_source, d_target.neuron, s.delay, f_addr);
	}
}



size_t
ConnectivityMatrix::createFcm(
		const network::Generator& net,
		const Mapper& mapper,
		WarpAddressTable& wtable,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
{
	size_t nextFreeWarp = 1; // leave space for null warp at beginning

	for(network::synapse_iterator s = net.synapse_begin();
			s != net.synapse_end(); ++s) {
		addSynapse(*s, mapper, nextFreeWarp, wtable, h_targets, h_weights);
	}
	return nextFreeWarp;
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
		const std::vector<weight_dt>& h_weights,
		bool logging)
{
	md_fcmPlaneSize = totalWarps * WARP_SIZE;
	size_t bytes = md_fcmPlaneSize * 2 * sizeof(synapse_t);

	void* d_fcm;
	d_malloc(&d_fcm, bytes, "fcm");
	md_fcm = boost::shared_ptr<synapse_t>(static_cast<synapse_t*>(d_fcm), d_free);
	md_fcmAllocated = bytes;

	CUDA_SAFE_CALL(setFcmPlaneSize(md_fcmPlaneSize));

	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_ADDRESS, h_targets, md_fcmPlaneSize);
	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_WEIGHT,
			reinterpret_cast<const synapse_t*>(&h_weights[0]), md_fcmPlaneSize);
}



void
ConnectivityMatrix::moveRcmToDevice()
{
	for(rcm_t::const_iterator i = m_rsynapses.begin(); i != m_rsynapses.end(); ++i) {
		i->second->moveToDevice();
	}

	std::vector<size_t> rpitch = r_partitionPitch();

	CUDA_SAFE_CALL(
		configureReverseAddressing(
				&rpitch[0],
				&r_partitionAddress()[0],
				&r_partitionStdp()[0],
				&r_partitionFAddress()[0],
				rpitch.size())
	);
}


void
ConnectivityMatrix::printMemoryUsage(std::ostream& out) const
{
	const size_t MEGA = 1<<20;
	out << "Memory usage on device:\n";
	out << "\tforward matrix: " << (md_fcmAllocated / MEGA) << "MB\n";
	out << "\treverse matrix: " << (d_allocatedRCM() / MEGA) << "MB (" << m_rsynapses.size() << " groups)\n";
	out << "\tglobal queue: " << (m_gq.allocated() / MEGA) << "MB\n";
	out << "\toutgoing: " << (m_outgoing.allocated() / MEGA) << "MB\n" << std::endl;
}




/* Data used when user reads FCM back from device. These are indexed by
 * (global) synapse ids, and are thus filled in a random order. To populate
 * these in a single pass over the input, resize on insertion.  The synapse ids
 * are required to form a contigous range, so every element should be assigned
 * exactly once. */
void
ConnectivityMatrix::addAuxTerminal(const Synapse& s, size_t addr)
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
ConnectivityMatrix::syncWeights(cycle_t cycle, const std::vector<synapse_id>& synapses)
{
	if(cycle != m_lastWeightSync && !synapses.empty() && !mh_weights.empty()) {
		//! \todo refine this by only doing the minimal amount of copying
		memcpyFromDevice(reinterpret_cast<synapse_t*>(&mh_weights[0]),
					md_fcm.get() + FCM_WEIGHT * md_fcmPlaneSize,
					md_fcmPlaneSize);
		m_lastWeightSync = cycle;
	}
	return mh_weights;
}



const std::vector<float>&
ConnectivityMatrix::getWeights(cycle_t cycle, const std::vector<synapse_id>& synapses)
{
	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	m_queriedWeights.resize(synapses.size());
	const std::vector<weight_dt>& h_weights = syncWeights(cycle, synapses);
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses.at(i);
		size_t addr = m_cmAux[neuronIndex(id)][synapseIndex(id)].addr();
		weight_dt w = h_weights[addr];
		m_queriedWeights[i] = fx_toFloat(w, m_fractionalBits);;
	}
	return m_queriedWeights;
}



template<typename T>
const std::vector<T>&
getSynapseState(
		bool writeOnlySynapses,
		const std::vector<synapse_id>& synapses,
		const boost::unordered_map<nidx_t, std::deque<AxonTerminalAux> >& cm,
		std::const_mem_fun_ref_t<T, AxonTerminalAux> fun,
		std::vector<T>& out)
{
	using boost::format;

	if(writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	out.resize(synapses.size());
	for(size_t i = 0, i_end = synapses.size(); i != i_end; ++i) {
		synapse_id id = synapses.at(i);
		boost::unordered_map<nidx_t, std::deque<AxonTerminalAux> >::const_iterator it = cm.find(neuronIndex(id));
		if(it == cm.end()) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing neuron id (%u) in synapse query") % neuronIndex(id)));
		}
		out[i] = fun(it->second.at(synapseIndex(id)));
	}
	return out;
}



const std::vector<unsigned>&
ConnectivityMatrix::getTargets(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(m_writeOnlySynapses, synapses, m_cmAux,
			std::mem_fun_ref(&AxonTerminalAux::target), m_queriedTargets);
}


const std::vector<unsigned>&
ConnectivityMatrix::getDelays(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(m_writeOnlySynapses, synapses, m_cmAux,
			std::mem_fun_ref(&AxonTerminalAux::delay), m_queriedDelays);
}


const std::vector<unsigned char>&
ConnectivityMatrix::getPlastic(const std::vector<synapse_id>& synapses)
{
	return getSynapseState(m_writeOnlySynapses, synapses, m_cmAux,
			std::mem_fun_ref(&AxonTerminalAux::plastic), m_queriedPlastic);
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
		+ m_gq.allocated()
		+ m_outgoing.allocated();
}



/* Map function over vector of reverse synapse matrix */
template<typename T, class S>
std::vector<T>
mapDevicePointer(const std::map<pidx_t, S*>& vec, std::const_mem_fun_t<T, S> fun)
{
	/* Always fill this in with zeros. */
	std::vector<T> ret(MAX_PARTITION_COUNT, T(0));
	for(typename std::map<pidx_t, S*>::const_iterator i = vec.begin();
			i != vec.end(); ++i) {
		ret.at(i->first) = fun(i->second);
	}
	return ret;
}



std::vector<size_t>
ConnectivityMatrix::r_partitionPitch() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::pitch));
}



std::vector<uint32_t*>
ConnectivityMatrix::r_partitionAddress() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_address));
}



std::vector<weight_dt*>
ConnectivityMatrix::r_partitionStdp() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_stdp));
}



std::vector<uint32_t*>
ConnectivityMatrix::r_partitionFAddress() const
{
	return mapDevicePointer(m_rsynapses, std::mem_fun(&RSMatrix::d_faddress));
}



	} // end namespace cuda
} // end namespace nemo
