#include "Neurons.hpp"

namespace nemo {
	namespace cpu {


Neurons::Neurons(const nemo::network::Generator& net,
				unsigned type_id,
				RandomMapper<nidx_t>& mapper) :
	m_mapper(mapper),
	m_type(net.neuronType(type_id)),
	m_nParam(m_type.parameterCount()),
	m_nState(m_type.stateVarCount()),
	m_param(boost::extents[m_nParam][net.neuronCount()]),
	m_state(boost::extents[m_type.stateHistory()][m_nState][net.neuronCount()]),
	m_stateCurrent(0),
	m_size(0),
	m_rng(net.neuronCount()),
	m_fstim(net.neuronCount(), 0),
	m_plugin(m_type.name(), "cpu"),
	m_update_neurons((cpu_update_neurons_t*) m_plugin.function("cpu_update_neurons"))
{
	using namespace nemo::network;

	std::fill(m_state.data(), m_state.data() + m_state.num_elements(), 0.0f);

	unsigned base = mapper.typeBase(type_id);

	for(neuron_iterator i = net.neuron_begin(type_id), i_end = net.neuron_end(type_id);
			i != i_end; ++i) {

		unsigned g_idx = i->first;
		unsigned l_idx = base + m_size;
		mapper.insert(g_idx, l_idx);
		mapper.insertTypeMapping(l_idx, type_id);

		const Neuron& n = i->second;
		setLocal(l_idx, n.getParameters(), n.getState());

		m_size++;
	}

	nemo::initialiseRng(m_mapper.minLocalIdx(), m_mapper.maxLocalIdx(), m_rng);
}



void
Neurons::set(unsigned g_idx, unsigned nargs, const float args[])
{
	using boost::format;

	if(nargs != m_nParam + m_nState) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unexpected number of parameters/state variables when modifying neuron. Expected %u, found %u")
						% (m_nParam + m_nState) % nargs));
	}

	setLocal(m_mapper.localIdx(g_idx), args, args+m_nParam);
}


void
Neurons::setLocal(unsigned l_idx, const float param[], const float state[])
{
	for(unsigned i=0; i < m_nParam; ++i) {
		m_param[i][l_idx] = param[i];
	}
	for(unsigned i=0; i < m_nState; ++i) {
		m_state[0][i][l_idx] = state[i];
	}
}



float
Neurons::getState(unsigned g_idx, unsigned var) const
{
	return m_state[m_stateCurrent][stateIndex(var)][m_mapper.localIdx(g_idx)];
}



void
Neurons::setState(unsigned g_idx, unsigned var, float val)
{
	m_state[m_stateCurrent][stateIndex(var)][m_mapper.localIdx(g_idx)] = val;
}



void
Neurons::setFiringStimulus(const std::vector<unsigned>& fstim)
{
	for(std::vector<unsigned>::const_iterator i = fstim.begin();
			i != fstim.end(); ++i) {
		m_fstim.at(m_mapper.localIdx(*i)) = 1;
	}
}



void
Neurons::update(
		unsigned cycle,
		unsigned fbits,
		fix_t current[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* rcm)
{
	unsigned start = m_mapper.minLocalIdx();
	unsigned end = m_mapper.maxLocalIdx() + 1;

	assert_or_throw(end <= size(), "Invalid neuron range in CPU backend neuron update");

	m_stateCurrent = (cycle+1) % m_type.stateHistory();
	m_update_neurons(start, end, cycle,
			m_param.data(), m_param.strides()[0],
			m_state.data(), m_state.strides()[0], m_state.strides()[1],
			fbits,
			&m_fstim[0],
			&m_rng[0],
			current,
			recentFiring,
			fired,
			rcm);
}



unsigned
Neurons::stateIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_nState) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid state variable index %u") % i));
	}
	return i;
}



unsigned
Neurons::parameterIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_nParam) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid parameter index %u") % i));
	}
	return i;
}


	} // end namespace cpu
} // end namespace nemo
