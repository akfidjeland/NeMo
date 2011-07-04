#ifndef NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP
#define NEMO_CUDA_PLUGIN_NEURON_MODEL_HPP

/* Common API for CUDA neuron model plugins */

#include <cuda_runtime.h>
#include <nemo/internal_types.h>
#include <nemo/cuda/types.h>
#include <nemo/cuda/parameters.cu_h>
#include <nemo/cuda/rng.cu_h>
#include <nemo/cuda/rcm.cu_h>

#ifdef __cplusplus
extern "C" {
#endif

/*! Update the state of a group of neurons of the same type
 *
 * \param globalPartitionCount number of partitions in network
 * \param localPartitionCount number of partitions in this group
 * \param basePartition global index of the first partition in this group
 * \param d_valid
 * 		bit vector indicating the valid neurons. This is a vector for all
 * 		partitions for the current neuron type only.
 */
typedef cudaError_t cuda_update_neurons_t(
		cudaStream_t stream,
		unsigned cycle,
		unsigned globalPartitionCount,
		unsigned localPartitionCount,
		unsigned basePartition,
		unsigned* d_partitionSize,
		param_t* d_globalParameters,
		float* df_neuronParameters,
		float* df_neuronState,
		nrng_t d_nrng,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		rcm_dt* d_rcm);

#ifdef __cplusplus
}
#endif

#endif
