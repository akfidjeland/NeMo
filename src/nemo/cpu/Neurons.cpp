#include "Neurons.hpp"

#include <nemo/fixedpoint.hpp>


namespace nemo {
	namespace cpu {


Neurons::Neurons(const nemo::network::Generator& net) :
	m_type(net.neuronType()),
	m_nParam(m_type.f_nParam()),
	m_nState(m_type.f_nState()),
	m_param(boost::extents[m_nParam][net.neuronCount()]),
	m_state(boost::extents[m_nState][net.neuronCount()]),
	m_size(0),
	m_rng(net.neuronCount()),
	m_fstim(net.neuronCount(), 0),
	m_plugin(m_type.name(), "cpu"),
	m_update_neurons((cpu_update_neurons_t*) m_plugin.function("cpu_update_neurons"))
{
	using namespace nemo::network;

	for(neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {

		unsigned g_idx = i->first;
		unsigned l_idx = m_size;
		m_mapper.insert(g_idx, l_idx);

		const Neuron& n = i->second;
		setLocal(l_idx, n.f_getParameters(), n.f_getState());

		m_size++;
	}

	nemo::initialiseRng(m_mapper.minLocalIdx(), m_mapper.maxLocalIdx(), m_rng);
}



void
Neurons::setLocal(unsigned l_idx, const float fParam[], const float fState[])
{
	for(unsigned i=0; i < m_nParam; ++i) {
		m_param[i][l_idx] = fParam[i];
	}
	for(unsigned i=0; i < m_nState; ++i) {
		m_state[i][l_idx] = fState[i];
	}
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
Neurons::update(int start, int end,
		unsigned fbits,
		fix_t current[],
		uint64_t recentFiring[],
		unsigned fired[])
{
	if(0 > start || start > end || end > int(size())) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Invalid neuron range in CPU backend neuron update");
	}

	m_update_neurons(start, end,
			m_param.data(), m_param.strides()[0],
			m_state.data(), m_state.strides()[0],
			fbits,
			&m_fstim[0],
			&m_rng[0],
			current,
			recentFiring,
			fired);
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
