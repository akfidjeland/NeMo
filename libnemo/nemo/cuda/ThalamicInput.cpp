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

#include <nemo/NetworkImpl.hpp>
#include <nemo/types.hpp>

#include "thalamicInput.cu_h"
#include "Mapper.hpp"



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

	/* To ensure consistent results in a parallel/concurrent setting (e.g.
	 * MPI), we need maintain a fixed mapping from global neuron indices to RNG
	 * seeds. Using the same basis seed on each node and just skipping the
	 * initial values provides a straightforward method to achieve this. For
	 * very large networks other methods (e.g. splitting RNGs) might be more
	 * appropriate. */
	for(unsigned gidx=0; gidx < 4 * mapper.minHostIdx(); ++gidx) {
		seed();
	}

	//! \todo store sigma with NeuronParameters and make ThalamicInput purely RNG state
	/* The RNG state vector needs to be filled with initialisation data. Each
	 * RNG needs 4 32-bit words of seed data, with each thread having a
	 * diferent seed. */

	typedef std::map<nidx_t, nemo::Neuron<float> >::const_iterator it;
	it neurons_end = net.m_neurons.end();

	for(unsigned gidx = mapper.minHostIdx(), gidx_end = mapper.maxHostIdx();
			gidx <= gidx_end; ++gidx) {
		it neuron = net.m_neurons.find(gidx);
		if(neuron == neurons_end) {
			/* ensure consistent seeding. See above comment */
			seed(); seed(); seed(); seed();
		} else {
			float sigma = neuron->second.sigma;
			m_inUse |= sigma != 0.0f;
			DeviceIdx didx = mapper.deviceIdx(gidx);
			m_sigma.setNeuron(didx.partition, didx.neuron, sigma);
			for(size_t plane=0; plane < 4; ++plane) {
				m_rngState.setNeuron(didx.partition, didx.neuron, seed(), plane);
			}
		}
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
