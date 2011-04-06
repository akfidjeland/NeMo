#ifndef NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP
#define NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP

/* Common API for CUDA neuron model plugins */

#include <cuda.h>
#include <nemo/internal_types.h>

cudaError_t
update_neurons(
		cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		bool thalamicInputEnabled,
		param_t* d_globalParameters,
		float* df_neuronParameters,
		float* df_neuronState,
		unsigned* du_neuronState,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired);

#endif
