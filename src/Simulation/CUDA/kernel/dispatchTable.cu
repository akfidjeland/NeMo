#ifndef DISPATCH_TABLE_CU
#define DISPATCH_TABLE_CU

#include "kernel.cu_h"
#include "dispatchTable.cu_h"


/*! \return
 *		Word pitch of connectivity matrix block
 */
__device__
size_t
f_pitch(fcm_ref_t r)
{
	return (size_t) r.x;
}



__device__
uint32_t*
f_base(fcm_ref_t r)
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



/*! \return	size (in words) of one plane of the given connectivity matrix */
__device__
size_t
f_size(fcm_ref_t ref)
{
	return MAX_PARTITION_SIZE * f_pitch(ref);
}



/* \return
 * 		Address of the beginning of the addressing part of forward connectivity
 * 		matrix block.
 */
__device__
uint*
f_address(uint32_t* base, size_t pitch)
{
	return base + FCM_ADDRESS * MAX_PARTITION_SIZE * pitch;
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
f_weights(fcm_ref_t ref)
{
	return (float*) f_base(ref) + FCM_WEIGHT * MAX_PARTITION_SIZE * f_pitch(ref);
}



/* \return
 * 		Address of the beginning of the weights part of forward connectivity
 * 		matrix block.
 */
__device__
float*
f_weights(uint32_t* base, size_t pitch)
{
	return (float*) base + FCM_WEIGHT * MAX_PARTITION_SIZE * pitch;
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


texture<fcm_ref_t, 3, cudaReadModeElementType> tf_refs;



__device__
fcm_ref_t
getFCM(uint sourcePartition, uint targetPartition, uint delay0)
{
	return tex3D(tf_refs, (float) delay0, (float) targetPartition, (float) sourcePartition);
}



__host__
cudaArray*
f_setDispatchTable(
		size_t partitionCount,
		size_t delayCount,
		const std::vector<fcm_ref_t>& h_table)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<fcm_ref_t>();

	cudaArray* d_table;
	cudaExtent ext = make_cudaExtent(delayCount, partitionCount, partitionCount);
	CUDA_SAFE_CALL(cudaMalloc3DArray(&d_table, &channelDesc, ext));

	cudaMemcpy3DParms copyParams = {0};
	copyParams.extent = ext;
	copyParams.kind = cudaMemcpyHostToDevice;
	copyParams.dstArray = d_table;
	copyParams.srcPtr = make_cudaPitchedPtr(
			(void*)&h_table[0],
			ext.width * sizeof(fcm_ref_t), ext.width, ext.height);
	CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tf_refs.addressMode[0] = cudaAddressModeClamp;
	tf_refs.addressMode[1] = cudaAddressModeClamp;
	tf_refs.addressMode[2] = cudaAddressModeClamp;
	tf_refs.filterMode = cudaFilterModePoint;
	tf_refs.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(tf_refs, d_table, channelDesc));
	return d_table;
}

#endif
