#ifndef DISPATCH_TABLE_CU
#define DISPATCH_TABLE_CU

#include "kernel.cu_h"
#include "dispatchTable.cu_h"


/*! \return
 *		Word pitch of connectivity matrix block
 */
__device__
size_t
f0_pitch(fcm_ref_t r)
{
	return (size_t) r.x;
}



__device__
uint32_t*
f0_base(fcm_ref_t r)
{
#ifdef __DEVICE_EMULATION__
	uint64_t ptr = r.z;
	ptr <<= 32;
	ptr |= r.y;
	return (uint32_t*) ptr;
#else
	return (uint32_t*) r.y;
#endif
}





/*!
 * \param ref
 *		Reference to connectivity matrix block for a particular partition/delay
 * \return
 * 		Address of the beginning of the addressing part of connectivity matrix
 * 		block specified by \a ref
 */
__device__
uint*
f0_address(fcm_ref_t ref)
{
	return f0_base(ref) + FCM_ADDRESS * MAX_PARTITION_SIZE * f0_pitch(ref);
}



/*!
 * \param ref
 *		Reference to connectivity matrix block for a particular partition/delay
 * \return
 * 		Address of the beginning of the weights part of connectivity matrix
 * 		block specified by \a ref
 */
__device__
float*
f0_weights(fcm_ref_t ref)
{
	return (float*) f0_base(ref) + FCM_WEIGHT * MAX_PARTITION_SIZE * f0_pitch(ref);
}



__host__
fcm_ref_t
fcm_packReference(void* address, size_t pitch)
{
	assert(sizeof(address) <= sizeof(uint64_t));

	uint64_t ptr64 = (uint64_t) address;

#ifdef __DEVICE_EMULATION__
	uint32_t low = (uint32_t) (ptr64 & 0xffffffff);
	uint32_t high = (uint32_t) ((ptr64 >> 32) & 0xffffffff);
	return make_uint4((uint) pitch, (uint) low, (uint) high, 0);
#else
	const uint64_t MAX_ADDRESS = 4294967296LL; // on device
	assert(ptr64 < MAX_ADDRESS);
	return make_uint2((uint) pitch, (uint) ptr64);
#endif
}


texture<fcm_ref_t, 2, cudaReadModeElementType> tf0_refs;


__device__
fcm_ref_t
getFCM(uint partition, uint delay)
{
	return tex2D(tf0_refs, (float) delay, (float) partition);
}



 __host__
cudaArray*
f0_setDispatchTable(
		size_t partitionCount,
		size_t delayCount,
		const std::vector<fcm_ref_t>& h_table)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<fcm_ref_t>();

	size_t width = delayCount;
	size_t height = partitionCount;

	assert(h_table.size() == width * height);

	cudaArray* d_table;
	CUDA_SAFE_CALL(cudaMallocArray(&d_table, &channelDesc, width, height));

	size_t bytes = height * width * sizeof(fcm_ref_t);
	CUDA_SAFE_CALL(cudaMemcpyToArray(d_table, 0, 0, &h_table[0], bytes, cudaMemcpyHostToDevice));

	// set texture parameters
	tf0_refs.addressMode[0] = cudaAddressModeClamp;
	tf0_refs.addressMode[1] = cudaAddressModeClamp;
	tf0_refs.filterMode = cudaFilterModePoint;
	tf0_refs.normalized = false;

	CUDA_SAFE_CALL(cudaBindTextureToArray(tf0_refs, d_table, channelDesc));

	return d_table;
}

#endif
