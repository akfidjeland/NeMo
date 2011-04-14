#include "Neurons.hpp"

#include <nemo/fixedpoint.hpp>


namespace nemo {
	namespace cpu {


Neurons::Neurons(const nemo::network::Generator& net) :
	m_neurons(net.neuronType()),
	m_rng(net.neuronCount()),
	m_fstim(net.neuronCount(), 0)
{
	using namespace nemo::network;

	for(neuron_iterator i = net.neuron_begin(), i_end = net.neuron_end();
			i != i_end; ++i) {
		unsigned g_idx = i->first;
		const Neuron& n = i->second;
		size_t l_idx = m_neurons.add(n.f_getParameters(), n.f_getState());
		m_mapper.insert(g_idx, l_idx);
	}

	nemo::initialiseRng(m_mapper.minLocalIdx(), m_mapper.maxLocalIdx(), m_rng);
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
	if(0 > start || start > end || end > int(m_neurons.size())) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Invalid neuron range in CPU backend neuron update");
	}

	const float* a = m_neurons.getParameter(PARAM_A);
	const float* b = m_neurons.getParameter(PARAM_B);
	const float* c = m_neurons.getParameter(PARAM_C);
	const float* d = m_neurons.getParameter(PARAM_D);
	const float* sigma = m_neurons.getParameter(PARAM_SIGMA);
	float* u = m_neurons.getState(STATE_U);
	float* v = m_neurons.getState(STATE_V);

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


	} // end namespace cpu
} // end namespace nemo
