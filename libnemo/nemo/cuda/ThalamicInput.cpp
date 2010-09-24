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

#include <nemo/network/Generator.hpp>
#include <nemo/types.hpp>
#include <nemo/RNG.hpp>

#include "thalamicInput.cu_h"
#include "Mapper.hpp"



namespace nemo {
	namespace cuda {


ThalamicInput::ThalamicInput(
		const nemo::network::Generator& net,
		const Mapper& mapper) :
	m_rngState(mapper.partitionCount(), mapper.partitionSize(), true),
	m_sigma(mapper.partitionCount(), mapper.partitionSize(), true),
	m_inUse(false)
{
	std::vector<nemo::RNG> rngs(mapper.maxHostIdx() - mapper.minHostIdx() + 1);
	initialiseRng(mapper.minHostIdx(), mapper.maxHostIdx(), rngs);

	for(network::neuron_iterator i = net.neuron_begin(); i != net.neuron_end(); ++i) {
		DeviceIdx didx = mapper.deviceIdx(i->first);
		float sigma = i->second.sigma;
		m_inUse |= sigma != 0.0f;
		m_sigma.setNeuron(didx.partition, didx.neuron, sigma);
		for(unsigned plane = 0; plane < 4; ++plane) {
			nidx_t localIdx = mapper.hostIdx(didx) - mapper.minHostIdx();
			m_rngState.setNeuron(didx.partition, didx.neuron, rngs[localIdx][plane], plane);
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
