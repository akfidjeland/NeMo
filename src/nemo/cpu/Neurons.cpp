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
	m_fstim(net.neuronCount(), 0)
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


#define SUBSTEPS 4
#define SUBSTEP_MULT 0.25

//! \todo share symbolic names with CUDA Izhikevich plugin
#define PARAM_A 0
#define PARAM_B 1
#define PARAM_C 2
#define PARAM_D 3
#define PARAM_SIGMA 4 // for gaussian RNG

#define STATE_U 0
#define STATE_V 1



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

	const float* a = parameterArray(PARAM_A);
	const float* b = parameterArray(PARAM_B);
	const float* c = parameterArray(PARAM_C);
	const float* d = parameterArray(PARAM_D);
	const float* sigma = parameterArray(PARAM_SIGMA);

	float* u = stateArray(STATE_U);
	float* v = stateArray(STATE_V);

	for(int n=start; n < end; n++) {

		float I = fx_toFloat(current[n], fbits);
		current[n] = 0;

		if(sigma[n] != 0.0f) {
			I += sigma[n] * (float) m_rng[n].gaussian();
		}

		fired[n] = 0;

		for(unsigned int t=0; t<SUBSTEPS; ++t) {
			if(!fired[n]) {
				v[n] += SUBSTEP_MULT * ((0.04* v[n] + 5.0) * v[n] + 140.0 - u[n] + I);
				u[n] += SUBSTEP_MULT * (a[n] * (b[n] * v[n] - u[n]));
				fired[n] = v[n] >= 30.0;
			}
		}

		fired[n] |= m_fstim[n];
		m_fstim[n] = 0;
		recentFiring[n] = (recentFiring[n] << 1) | (uint64_t) fired[n];

		if(fired[n]) {
			v[n] = c[n];
			u[n] += d[n];
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

	}
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



const float*
Neurons::parameterArray(unsigned pidx) const
{
	const param_type::index* strides = m_param.strides();
	return m_param.data() + pidx * strides[0];
}


float*
Neurons::stateArray(unsigned sidx)
{
	const param_type::index* strides = m_state.strides();
	//const param_type::size_type* shape m_state.shape();
	return m_state.data() + sidx * strides[0];
}

	} // end namespace cpu
} // end namespace nemo
