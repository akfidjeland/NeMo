#ifndef NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP
#define NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP

/* Common API for CUDA neuron model plugins */

#include <cuda_runtime.h>
#include <nemo/internal_types.h>
#include <nemo/cuda/types.h>
#include <nemo/cuda/parameters.cu_h>

#ifdef __cplusplus
extern "C" {
#endif

typedef cudaError_t update_neurons_t(
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

#ifdef __cplusplus
}
#endif

#endif
