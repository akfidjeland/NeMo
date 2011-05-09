/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <iostream>

#include <boost/tuple/tuple_comparison.hpp>

#include <nemo/ConfigurationImpl.hpp>
#include <nemo/NeuronType.hpp>
#include <nemo/cuda/kernel.cu_h>
#include <nemo/cuda/rcm.cu_h>

#include "RCM.hpp"


namespace boost {
	namespace tuples {


//! \todo share the hashing code with FcmIndex
template<typename T1, typename T2>
std::size_t
hash_value(const tuple<T1, T2>& k)
{
	std::size_t seed = 0;
	boost::hash_combine(seed, boost::tuples::get<0>(k));
	boost::hash_combine(seed, boost::tuples::get<1>(k));
	return seed;
}

	} // end namespace tuples
} // end namespace boost


namespace nemo {
	namespace cuda {
		namespace construction {


RCM::RCM(const nemo::ConfigurationImpl& conf, const nemo::NeuronType& type) :
	/* leave space for null warp at beginning */
	m_synapseCount(0),
	m_nextFreeWarp(1),
	m_data(WARP_SIZE, INVALID_REVERSE_SYNAPSE),
	m_useData(type.usesRcmSources() || type.usesRcmDelays() || conf.stdpFunction()),
	m_forward(WARP_SIZE, 0),
	m_useForward(type.usesRcmForward() || conf.stdpFunction()),
	m_useWeights(type.usesRcmWeights()),
	m_enabled(m_useData || m_useForward || m_useWeights),
	m_stdpEnabled(conf.stdpFunction())
{
	/* The approach to setting up the reverse connectivity is a bit messy in
	 * the current version. Really, the STDP configuration ought to be a part
	 * of the neuron model, whereas STDP is a completely separate issue.
	 *
	 * The problem is that when STDP is enabled, we support a mix of static and
	 * dynamic synapses (specified when adding synapses). When a neuron model
	 * calls for the RCM to be present (in order to get access to presynaptic
	 * state), however, we most likely want /all/ synapses to be present in the
	 * RCM.
	 *
	 * We currently support /either/ the plastic-only RCM (if STDP is enabled)
	 * or the all-synapse RCM (if the neuron model calls for the RCM to be
	 * present).
	 */
	bool typeRequiresRcm =
			   type.usesRcmSources()
			|| type.usesRcmDelays()
			|| type.usesRcmForward()
			|| type.usesRcmWeights();
	if(typeRequiresRcm && conf.stdpFunction()) {
		throw nemo::exception(NEMO_API_UNSUPPORTED,
				"The current version does not support a mix of STDP with neuron types which require reverse connectivity for other purposes");
	}
}



/*! Allocate space for a new RCM synapse for the given (target) neuron.
 *
 * \return
 * 		word offset for the synapse. This is the same for all the different
 * 		planes of data.
 */
size_t
RCM::allocateSynapse(const DeviceIdx& target)
{
	m_synapseCount += 1;

	key k(target.partition, target.neuron);
	unsigned& dataRowLength = m_dataRowLength[k];
	unsigned column = dataRowLength % WARP_SIZE;
	dataRowLength += 1;

	std::vector<size_t>& warps = m_warps[k];

	size_t row;
	if(column == 0) {
		/* Add synapse to a new warp */
		warps.push_back(m_nextFreeWarp);
		row = m_nextFreeWarp;
		m_nextFreeWarp += 1;
		/* Resize host buffers to accomodate the new warp. This allocation
		 * scheme could potentially result in a large number of reallocations,
		 * so we might be better off allocating larger chunks here */
		size_t size = m_nextFreeWarp * WARP_SIZE;
		if(m_useData) {
			m_data.resize(size, INVALID_REVERSE_SYNAPSE);
		}
		if(m_useForward) {
			m_forward.resize(size, 0);
		}
		if(m_useWeights) {
			m_weights.resize(size, 0.0f);
		}
	} else {
		/* Add synapse to an existing partially-filled warp */
		row = *warps.rbegin();
	}

	return row * WARP_SIZE + column;
}



void
RCM::addSynapse(
		const Synapse& s,
		const DeviceIdx& d_source,
		const DeviceIdx& d_target,
		size_t f_addr)
{
	if(m_enabled) {
		if(!m_stdpEnabled || s.plastic()) {
			size_t r_addr = allocateSynapse(d_target);
			if(m_useData) {
				m_data[r_addr] = make_rsynapse(d_source.partition, d_source.neuron, s.delay);
			}
			if(m_useForward) {
				m_forward[r_addr] = f_addr;
			}
			if(m_useWeights) {
				m_weights[r_addr] = s.weight();
			}
		}
	}
}


size_t
RCM::size() const
{
	return m_nextFreeWarp * WARP_SIZE;
}

		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo
