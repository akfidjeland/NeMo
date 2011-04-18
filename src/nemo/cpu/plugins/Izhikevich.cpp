#include <nemo/fixedpoint.hpp>

#include "neuron_model.h"
#include "Izhikevich.hpp"


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
update_neurons(
		int start, int end,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateStride,
		unsigned fbits,
		unsigned fstim[],
		nemo::RNG rng[],
		fix_t current[],
		uint64_t recentFiring[],
		unsigned fired[])
{
	const float* a = paramBase + PARAM_A * paramStride;
	const float* b = paramBase + PARAM_B * paramStride;
	const float* c = paramBase + PARAM_C * paramStride;
	const float* d = paramBase + PARAM_D * paramStride;
	const float* sigma = paramBase + PARAM_SIGMA * paramStride;

	float* u = stateBase + STATE_U * stateStride;
	float* v = stateBase + STATE_V * stateStride;

	for(int n=start; n < end; n++) {

		float I = fx_toFloat(current[n], fbits);
		current[n] = 0;

		if(sigma[n] != 0.0f) {
			I += sigma[n] * (float) rng[n].gaussian();
		}

		fired[n] = 0;

		for(unsigned int t=0; t<SUBSTEPS; ++t) {
			if(!fired[n]) {
				v[n] += SUBSTEP_MULT * ((0.04* v[n] + 5.0) * v[n] + 140.0 - u[n] + I);
				u[n] += SUBSTEP_MULT * (a[n] * (b[n] * v[n] - u[n]));
				fired[n] = v[n] >= 30.0;
			}
		}

		fired[n] |= fstim[n];
		fstim[n] = 0;
		recentFiring[n] = (recentFiring[n] << 1) | (uint64_t) fired[n];

		if(fired[n]) {
			v[n] = c[n];
			u[n] += d[n];
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}
	}

}
