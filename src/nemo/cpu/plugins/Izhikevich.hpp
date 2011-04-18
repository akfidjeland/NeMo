#ifndef NEMO_CPU_PLUGINS_IZHIKEVICH_HPP
#define NEMO_CPU_PLUGINS_IZHIKEVICH_HPP

#include "neuron_model.h"

#ifdef __cplusplus
extern "C" {
#endif

void
update_neurons(
		int start, int end,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateStride,
		unsigned fbits,
		unsigned fstim[],
		RNG rng[],
		fix_t current[],
		uint64_t recentFiring[],
		unsigned fired[]);

#ifdef __cplusplus
}
#endif

#endif
