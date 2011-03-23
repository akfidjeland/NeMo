/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Neurons.hpp"

#include <vector>

#include <boost/format.hpp>

#include <nemo/network/Generator.hpp>
#include <nemo/RNG.hpp>

#include "types.h"
#include "exception.hpp"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


Neurons::Neurons(const network::Generator& net, Mapper& mapper) :
	//! \todo set these sizes based on configuration
	//! \todo: remove these here. Handle inside NVector instead
	mf_nParams(NEURON_FLOAT_PARAM_COUNT),
	mf_nStateVars(NEURON_FLOAT_STATE_COUNT),
	mf_param(NEURON_FLOAT_PARAM_COUNT, mapper.partitionCount(), mapper.partitionSize(), true, false),
	mf_state(NEURON_FLOAT_STATE_COUNT, mapper.partitionCount(), mapper.partitionSize(), true, false),
	mu_state(NEURON_UNSIGNED_STATE_COUNT, mapper.partitionCount(), mapper.partitionSize(), true, false),
	m_valid(mapper.partitionCount(), true),
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
		const nemo::Neuron& n = i->second;

		for(unsigned i=0; i < mf_nParams; ++i) {
			mf_param.setNeuron(dev.partition, dev.neuron, n.f_getParameter(i), i);
		}
		for(unsigned i=0; i < mf_nStateVars; ++i) {
			mf_state.setNeuron(dev.partition, dev.neuron, n.f_getState(i), i);
		}

		m_valid.setNeuron(dev);

		float sigma = n.f_getParameter(PARAM_SIGMA);
		m_rngEnabled |= sigma != 0.0f;
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
	m_valid.moveToDevice();
	configurePartitionSizes(maxPartitionNeuron);
}



size_t
Neurons::d_allocated() const
{
	return mf_param.d_allocated()
		+ mf_state.d_allocated()
		+ mu_state.d_allocated();
}


void
Neurons::configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron)
{
	std::vector<unsigned> partitionSizes(MAX_PARTITION_COUNT, 0);
	for(std::map<pidx_t, nidx_t>::const_iterator i = maxPartitionNeuron.begin();
			i != maxPartitionNeuron.end(); ++i) {
		partitionSizes.at(i->first) = i->second + 1;
	}
	CUDA_SAFE_CALL(configurePartitionSize(&partitionSizes[0], partitionSizes.size()));
}



size_t
Neurons::wordPitch32() const
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
Neurons::step(cycle_t cycle)
{
	if(mf_paramDirty) {
		mf_param.copyToDevice();
		mf_paramDirty = false;
	}
	if(mf_stateDirty) {
		mf_state.copyToDevice();
		mf_stateDirty = false;
	}
	m_cycle = cycle;
}



inline
void
verifyParameterIndex(unsigned parameter, unsigned maxParameter)
{
	using boost::format;
	if(parameter >= maxParameter) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % parameter));
	}
}



float
Neurons::getParameter(const DeviceIdx& idx, unsigned parameter) const
{
	verifyParameterIndex(parameter, mf_nParams);
	return mf_param.getNeuron(idx.partition, idx.neuron, parameter);
}



void
Neurons::setParameter(const DeviceIdx& idx, unsigned parameter, float value)
{
	verifyParameterIndex(parameter, mf_nParams);
	mf_param.setNeuron(idx.partition, idx.neuron, value, parameter);
	mf_paramDirty = true;
}



void
Neurons::readStateFromDevice() const
{
	if(mf_lastSync != m_cycle) {
		mf_state.copyFromDevice();
		mf_lastSync = m_cycle;
	}
}



inline
void
verifyStateVariableIndex(unsigned var, unsigned maxVar)
{
	using boost::format;
	if(var >= maxVar) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % var));
	}
}


float
Neurons::getState(const DeviceIdx& idx, unsigned var) const
{
	verifyStateVariableIndex(var, mf_nStateVars);
	readStateFromDevice();
	return mf_state.getNeuron(idx.partition, idx.neuron, var);
}



void
Neurons::setState(const DeviceIdx& idx, unsigned var, float value)
{
	verifyStateVariableIndex(var, mf_nStateVars);
	readStateFromDevice();
	mf_state.setNeuron(idx.partition, idx.neuron, value, var);
	mf_stateDirty = true;
}


	} // end namespace cuda
} // end namespace nemo
