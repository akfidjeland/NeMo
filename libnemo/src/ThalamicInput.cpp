//! \file ThalamicInput.cpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ThalamicInput.hpp"

#include <map>
#include <boost/random.hpp>

#include "thalamicInput.cu_h"
#include "DeviceIdx.hpp"
#include "NetworkImpl.hpp"
#include <types.hpp>



namespace nemo {
	namespace cuda {


ThalamicInput::ThalamicInput(
		const nemo::NetworkImpl& net,
		const Mapper& mapper) :
	m_rngState(mapper.partitionCount(), mapper.partitionSize(), true, 4),
	m_sigma(mapper.partitionCount(), mapper.partitionSize(), true, 1),
	m_inUse(false)
{
	//! \todo allow users to seed this RNG
	typedef boost::mt19937 rng_t;
	rng_t rng;

	boost::variate_generator<rng_t, boost::uniform_int<unsigned long> >
		seed(rng, boost::uniform_int<unsigned long>(0, 0x7fffffff));
	//! \todo store sigma with NeuronParameters and make ThalamicInput purely RNG state
	/* The RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data, with each thread having a
	 * diferent seed. */
	for(std::map<nidx_t, nemo::Neuron<float> >::const_iterator i = net.m_neurons.begin();
			i != net.m_neurons.end(); ++i) {
		DeviceIdx didx = mapper.deviceIdx(i->first);
		float val = i->second.sigma;
		m_inUse = val != 0.0f;
		m_sigma.setNeuron(didx.partition, didx.neuron, val);
		m_rngState.setNeuron(didx.partition, didx.neuron, seed());
	}

	if(m_inUse) {
		m_rngState.moveToDevice();
		m_sigma.moveToDevice();
	}
}



size_t
ThalamicInput::d_allocated() const
{
	return m_rngState.d_allocated() + m_sigma.d_allocated();
}



unsigned* 
ThalamicInput::deviceRngState() const
{
	return m_inUse ? m_rngState.deviceData() : NULL;
}



float* 
ThalamicInput::deviceSigma() const
{
	return m_inUse ? m_sigma.deviceData() : NULL;
}



size_t
ThalamicInput::wordPitch() const
{
	size_t p1 = m_rngState.wordPitch();
	size_t p2 = m_sigma.wordPitch();
	//! \todo throw exception here instead
	assert(p1 == p2);
	return p1 == p2 ? p1 : 0;
}

	} // end namespace cuda
} // end namespace nemo
