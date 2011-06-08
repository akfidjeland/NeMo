#ifndef NEMO_CPU_PLUGIN_NEURON_MODEL_H
#define NEMO_CPU_PLUGIN_NEURON_MODEL_H

/* Common API for CPU neuron model plugins */

#include <nemo/internal_types.h>
#include <nemo/RNG.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/* Update a number of neurons in a contigous range */
typedef void cpu_update_neurons_t(
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
