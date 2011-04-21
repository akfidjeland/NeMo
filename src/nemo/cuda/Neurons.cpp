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

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include <nemo/RNG.hpp>

#include "types.h"
#include "exception.hpp"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


Neurons::Neurons(const network::Generator& net, const mapper_type& mapper) :
	m_mapper(mapper),
	m_type(net.neuronType()),
	mf_param(m_type.f_nParam(), mapper.partitionCount(), mapper.partitionSize(), true, false),
	mf_state(m_type.f_nState() * m_type.stateHistory(),
			mapper.partitionCount(), mapper.partitionSize(), true, false),
	m_stateCurrent(0),
	m_valid(mapper.partitionCount(), true),
	m_cycle(~0),
	mf_lastSync(~0),
	mf_paramDirty(false),
	mf_stateDirty(false),
	m_plugin(m_type.name(), "cuda"),
	m_update_neurons((update_neurons_t*) m_plugin.function("update_neurons"))
{

	if(m_type.usesNormalRNG()) {
		m_nrngState = NVector<unsigned>(RNG_STATE_COUNT,
				mapper.partitionCount(), mapper.partitionSize(), true, false);
	}

	std::map<pidx_t, nidx_t> maxPartitionNeuron;

	/* Create all the RNG seeds */
	//! \todo seed this from configuration
	std::vector<RNG> rngs(mapper.maxHandledGlobalIdx() - mapper.minHandledGlobalIdx() + 1);
	initialiseRng(mapper.minHandledGlobalIdx(), mapper.maxHandledGlobalIdx(), rngs);

	for(network::neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {

		//! \todo insertion here, but make sure the usage is correct in the Simulation class
		DeviceIdx dev = mapper.localIdx(i->first);
		const nemo::Neuron& n = i->second;

		for(unsigned i=0, i_end=parameterCount(); i < i_end; ++i) {
			mf_param.setNeuron(dev.partition, dev.neuron, n.f_getParameter(i), i);
		}
		for(unsigned i=0, i_end=stateVarCount(); i < i_end; ++i) {
			// no need to offset based on time here, since this is beginning of the simulation.
			mf_state.setNeuron(dev.partition, dev.neuron, n.f_getState(i), i);
		}

		m_valid.setNeuron(dev);

		nidx_t localIdx = mapper.globalIdx(dev) - mapper.minHandledGlobalIdx();
		for(unsigned plane = 0; plane < 4; ++plane) {
			m_nrngState.setNeuron(dev.partition, dev.neuron, rngs[localIdx].state[plane], plane);
		}

		maxPartitionNeuron[dev.partition] =
			std::max(maxPartitionNeuron[dev.partition], dev.neuron);
	}

	mf_param.copyToDevice();
	mf_state.copyToDevice();
	m_nrngState.moveToDevice();
	m_nrng.state = m_nrngState.deviceData();
	m_nrng.pitch = m_nrngState.wordPitch();
	m_valid.moveToDevice();
	configurePartitionSizes(maxPartitionNeuron);
}



size_t
Neurons::d_allocated() const
{
	return mf_param.d_allocated()
		+ mf_state.d_allocated()
		+ m_nrngState.d_allocated();
}


void
Neurons::configurePartitionSizes(const std::map<pidx_t, nidx_t>& maxPartitionNeuron)
{
	md_partitionSize = d_array<unsigned>(MAX_PARTITION_COUNT, "partition size array");
	std::vector<unsigned> h_partitionSize(MAX_PARTITION_COUNT, 0);
	for(std::map<pidx_t, nidx_t>::const_iterator i = maxPartitionNeuron.begin();
			i != maxPartitionNeuron.end(); ++i) {
		h_partitionSize.at(i->first) = i->second + 1;
	}
	memcpyToDevice(md_partitionSize.get(), h_partitionSize);
}



size_t
Neurons::wordPitch32() const
{
	size_t f_param_pitch = mf_param.wordPitch();
	size_t f_state_pitch = mf_state.wordPitch();
	if(f_param_pitch != f_state_pitch) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "State and parameter data have different pitches");
	}
	return f_param_pitch;
}



cudaError_t
Neurons::update(
		cudaStream_t stream,
		cycle_t cycle,
		param_t* d_params,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired)
{
	syncToDevice();
	m_cycle = cycle;
	m_stateCurrent = (cycle+1) % m_type.stateHistory();
	return m_update_neurons(stream,
			cycle,
			m_mapper.partitionCount(),
			md_partitionSize.get(),
			d_params,
			mf_param.deviceData(),
			mf_state.deviceData(),
			m_nrng,
			m_valid.d_data(),
			d_fstim, d_istim,
			d_current, d_fout, d_nFired, d_fired);
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
	verifyParameterIndex(parameter, parameterCount());
	return mf_param.getNeuron(idx.partition, idx.neuron, parameter);
}



size_t
Neurons::currentStateVariable(unsigned var) const
{
	return m_stateCurrent * stateVarCount() + var;
}



void
Neurons::setNeuron(const DeviceIdx& dev, const float param[], const float state[])
{
	readStateFromDevice();
	for(unsigned i=0, i_end=parameterCount(); i < i_end; ++i) {
		mf_param.setNeuron(dev.partition, dev.neuron, param[i], i);
	}
	mf_paramDirty = true;
	for(unsigned i=0, i_end=stateVarCount(); i < i_end; ++i) {
		mf_state.setNeuron(dev.partition, dev.neuron, state[i], currentStateVariable(i));
	}
	mf_stateDirty = true;
}



void
Neurons::setParameter(const DeviceIdx& idx, unsigned parameter, float value)
{
	verifyParameterIndex(parameter, parameterCount());
	mf_param.setNeuron(idx.partition, idx.neuron, value, parameter);
	mf_paramDirty = true;
}



void
Neurons::readStateFromDevice() const
{
	if(mf_lastSync != m_cycle) {
		//! \todo read only part of the data here
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
Neurons::getMembranePotential(const DeviceIdx& neuron) const
{
	return getState(neuron, m_type.membranePotential());
}



float
Neurons::getState(const DeviceIdx& idx, unsigned var) const
{
	verifyStateVariableIndex(var, stateVarCount());
	readStateFromDevice();
	return mf_state.getNeuron(idx.partition, idx.neuron, currentStateVariable(var));
}



void
Neurons::setState(const DeviceIdx& idx, unsigned var, float value)
{
	verifyStateVariableIndex(var, stateVarCount());
	readStateFromDevice();
	mf_state.setNeuron(idx.partition, idx.neuron, value, currentStateVariable(var));
	mf_stateDirty = true;
}



void
Neurons::syncToDevice()
{
	if(mf_paramDirty) {
		mf_param.copyToDevice();
		mf_paramDirty = false;
	}
	if(mf_stateDirty) {
		mf_state.copyToDevice();
		mf_stateDirty = false;
	}
}

	} // end namespace cuda
} // end namespace nemo
