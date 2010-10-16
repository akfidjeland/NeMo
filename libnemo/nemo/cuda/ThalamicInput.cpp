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
	m_inUse(false)
{
	std::vector<nemo::RNG> rngs(mapper.maxHandledGlobalIdx() - mapper.minHandledGlobalIdx() + 1);
	initialiseRng(mapper.minHandledGlobalIdx(), mapper.maxHandledGlobalIdx(), rngs);

	for(network::neuron_iterator i = net.neuron_begin(); i != net.neuron_end(); ++i) {
		DeviceIdx didx = mapper.deviceIdx(i->first);
		float sigma = i->second.sigma;
		m_inUse |= sigma != 0.0f;
		for(unsigned plane = 0; plane < 4; ++plane) {
			nidx_t localIdx = mapper.globalIdx(didx) - mapper.minHandledGlobalIdx();
			m_rngState.setNeuron(didx.partition, didx.neuron, rngs[localIdx][plane], plane);
		}
	}

	if(m_inUse) {
		m_rngState.moveToDevice();
	}
}



size_t
ThalamicInput::d_allocated() const
{
	return m_rngState.d_allocated();
}



unsigned* 
ThalamicInput::deviceRngState() const
{
	return m_inUse ? m_rngState.deviceData() : NULL;
}



size_t
ThalamicInput::wordPitch() const
{
	return m_rngState.wordPitch();
}


	} // end namespace cuda
} // end namespace nemo
