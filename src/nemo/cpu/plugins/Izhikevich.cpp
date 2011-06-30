#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/Izhikevich.h>

#include "neuron_model.h"

const unsigned SUBSTEPS = 4;
const float SUBSTEP_MULT = 0.25f;


extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		int start, int end,
		unsigned cycle,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		unsigned fbits,
		unsigned fstim[],
		RNG rng[],
		fix_t current[],
		uint64_t recentFiring[],
		unsigned fired[])
{
	const float* a = paramBase + PARAM_A * paramStride;
	const float* b = paramBase + PARAM_B * paramStride;
	const float* c = paramBase + PARAM_C * paramStride;
	const float* d = paramBase + PARAM_D * paramStride;
	const float* sigma = paramBase + PARAM_SIGMA * paramStride;

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;
	const float* u0 = stateBase + b0 * stateHistoryStride + STATE_U * stateVarStride;
	const float* v0 = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* u1 = stateBase + b1 * stateHistoryStride + STATE_U * stateVarStride;
	float* v1 = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;

	for(int n=start; n < end; n++) {

		float I = fx_toFloat(current[n], fbits);
		current[n] = 0;

		if(sigma[n] != 0.0f) {
			I += sigma[n] * nrand(&rng[n]);
		}

		fired[n] = 0;

		float u = u0[n];
		float v = v0[n];

		for(unsigned t=0; t<SUBSTEPS; ++t) {
			if(!fired[n]) {
				v += SUBSTEP_MULT * ((0.04* v + 5.0) * v + 140.0- u + I);
				u += SUBSTEP_MULT * (a[n] * (b[n] * v - u));
				fired[n] = v >= 30.0;
			}
		}

		fired[n] |= fstim[n];
		fstim[n] = 0;
		recentFiring[n] = (recentFiring[n] << 1) | (uint64_t) fired[n];

		if(fired[n]) {
			v = c[n];
			u += d[n];
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

		u1[n] = u;
		v1[n] = v;
	}
}
