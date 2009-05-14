#ifndef IZHIKEVICH_KERNEL_H
#define IZHIKEVICH_KERNEL_H

#include <stdint.h>

/* Error codes */
enum KernelError {
	KERNEL_OK,
	KERNEL_INVALID_PROBE,
	KERNEL_INVALID_DEVICE_MEMORY,
	KERNEL_MAX_CLUSTERS_EXCEEDED,
	KERNEL_CONSTANT_MEMORY_ERROR,
	KERNEL_INSUFFICIENT_SHARED_MEMORY,
	KERNEL_CUDA_ERROR
};

//#define BIT_PACK_DELAYS

#define MAX_THREAD_BLOCKS 512

//! \todo should change this to 64
#define MAX_DELAY 32

#define DENSE_ENCODING -1

/* kernel wrapper.
 *
 * \param updates 
 * 		Number of neuron update steps to do for this kernel invocation. A
 * 		larger value means higher performance due to less invocation overhead,
 * 		but at the cost of delayed spike delivery.
 * \param gMem
 * 		Device memory pointers
 * \param extI
 * 		Vector of external stimuli, one per neuron
 * \param firing
 * 		Vector of returned firing neurons for a single cluster
 * \param probe
 * 		Index of cluster to probe for firing information
 */
KernelError 
step(struct cudaDeviceProp* deviceProperties,
		int currentCycle,
		int udpates,
		class DeviceMemory* gMem,
		float currentScaling,
		const float* extI,
		const uint32_t* extFiring,
		int* spikes,
		float* v,
		int probe);



/*! Allocate constant memory for cluster configuration data, and copy this from
 * the host to the device. 
 *
 * \param externalCurrent
 * 		Array of length MAX_THREAD_BLOCKS of flags indicating whether each
 * 		cluster receives an external input current.	
 * \param externalFiring
 * 		Array of length MAX_THREAD_BLOCKS of flags indicating whether each 
 * 		cluster can be forced to fire by the host 
 * \param maxColumnIndex
 * 		Array of length MAX_THREAD_BLOCKS specifying the maximum column index
 * 		among all the rows for each cluster. If the encoding is dense, this 
 * 		value is set to DENSE_ENCODING.
 */
KernelError
configureClusters(const char* externalCurrent,
		const char* externalFiring,
		const int* maxColumnIndex);

#endif
