/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "NeuronParameters.hpp"

#include <vector>

#include <nemo/network/Generator.hpp>
#include <nemo/RNG.hpp>

#include "types.h"
#include "exception.hpp"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


NeuronParameters::NeuronParameters(const network::Generator& net, Mapper& mapper) :
	mf_param(mapper.partitionCount(), mapper.partitionSize(), true, false),
	mf_state(mapper.partitionCount(), mapper.partitionSize(), true, false),
	mu_state(mapper.partitionCount(), mapper.partitionSize(), true, false),
	m_cycle(0),
	mf_lastSync(~0),
	mf_paramDirty(false),
	mf_stateDirty(false),
	m_rngEnabled(false)
{
	std::map<pidx_t, nidx_t> maxPartitionNeuron;

	/* Create all the RNG seeds */
	//! \todo seed this from configuration
	std::vector<nemo::RNG> rngs(mapper.maxHandledGlobalIdx() - mapper.minHandledGlobalIdx() + 1);
	initialiseRng(mapper.minHandledGlobalIdx(), mapper.maxHandledGlobalIdx(), rngs);

	for(network::neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {

		DeviceIdx dev = mapper.addIdx(i->first);
		const nemo::Neuron<float>& n = i->second;

		mf_param.setNeuron(dev.partition, dev.neuron, n.a, PARAM_A);
		mf_param.setNeuron(dev.partition, dev.neuron, n.b, PARAM_B);
		mf_param.setNeuron(dev.partition, dev.neuron, n.c, PARAM_C);
		mf_param.setNeuron(dev.partition, dev.neuron, n.d, PARAM_D);
		mf_param.setNeuron(dev.partition, dev.neuron, n.sigma, PARAM_SIGMA);
		mf_state.setNeuron(dev.partition, dev.neuron, n.u, STATE_U);
		mf_state.setNeuron(dev.partition, dev.neuron, n.v, STATE_V);

		m_rngEnabled |= n.sigma != 0.0f;
		nidx_t localIdx = mapper.globalIdx(dev) - mapper.minHandledGlobalIdx();
		for(unsigned plane = 0; plane < 4; ++plane) {
			mu_state.setNeuron(dev.partition, dev.neuron, rngs[localIdx][plane], STATE_RNG+plane);
		}

		maxPartitionNeuron[dev.partition] =
			std::max(maxPartitionNeuron[dev.partition], dev.neuron);
	}

	mf_param.copyToDevice();
	mf_state.copyToDevice();
	mu_state.moveToDevice();
	configurePartitionSizes(maxPartitionNeuron);
}



size_t
NeuronParameters::d_allocated() const
{
	return mf_param.d_allocated()
		+ mf_state.d_allocated()
		+ mu_state.d_allocated();
}


void
NeuronParameters::configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron)
{
	if(maxPartitionNeuron.size() == 0) {
		return;
	}

	size_t maxPidx = maxPartitionNeuron.rbegin()->first;
	std::vector<unsigned> partitionSizes(maxPidx+1, 0);

	for(std::map<pidx_t, nidx_t>::const_iterator i = maxPartitionNeuron.begin();
			i != maxPartitionNeuron.end(); ++i) {
		partitionSizes.at(i->first) = i->second + 1;
	}

	CUDA_SAFE_CALL(configurePartitionSize(&partitionSizes[0], partitionSizes.size()));
}



size_t
NeuronParameters::wordPitch() const
{
	size_t f_param_pitch = mf_param.wordPitch();
	size_t f_state_pitch = mf_state.wordPitch();
	size_t u_state_pitch = mu_state.wordPitch();
	if(f_param_pitch != f_state_pitch || f_param_pitch != u_state_pitch) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "State and parameter data have different pitches");
	}
	return f_param_pitch;
}



void
NeuronParameters::step(cycle_t cycle)
{
	if(mf_paramDirty) {
		mf_param.copyToDevice();
	}
	if(mf_stateDirty) {
		mf_state.copyToDevice();
	}
	m_cycle = cycle;
}



float
NeuronParameters::getParameter(const DeviceIdx& idx, int parameter) const
{
	return mf_param.getNeuron(idx.partition, idx.neuron, parameter);
}



void
NeuronParameters::setParameter(const DeviceIdx& idx, int parameter, float value)
{
	mf_param.setNeuron(idx.partition, idx.neuron, value, parameter);
	mf_paramDirty = true;
}



void
NeuronParameters::readStateFromDevice() const
{
	if(mf_lastSync != m_cycle) {
		mf_state.copyFromDevice();
		mf_lastSync = m_cycle;
	}
}



float
NeuronParameters::getState(const DeviceIdx& idx, int parameter) const
{
	readStateFromDevice();
	return mf_param.getNeuron(idx.partition, idx.neuron, parameter);
}



void
NeuronParameters::setState(const DeviceIdx& idx, int parameter, float value)
{
	readStateFromDevice();
	mf_param.setNeuron(idx.partition, idx.neuron, value, parameter);
	mf_paramDirty = true;
}


	} // end namespace cuda
} // end namespace nemo
