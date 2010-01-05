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



/*! \return	size (in words) of one plane of the given connectivity matrix */
__device__
size_t
f0_size(fcm_ref_t r)
{
	return MAX_PARTITION_SIZE * f0_pitch(r);
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



//! \todo rename
__device__
uint*
f0_address2(uint32_t* base, size_t pitch)
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
f0_weights(fcm_ref_t ref)
{
	return (float*) f0_base(ref) + FCM_WEIGHT * MAX_PARTITION_SIZE * f0_pitch(ref);
}



__device__
float*
f0_weights2(uint32_t* base, size_t pitch)
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


texture<fcm_ref_t, 2, cudaReadModeElementType> tf0_refs;
texture<fcm_ref_t, 2, cudaReadModeElementType> tf1_refs;

//! \todo rename once we have removed old format
texture<fcm_ref_t, 3, cudaReadModeElementType> tf1_refs2;



/*! \param delay0 0-based delay (i.e. delay in ms - 1) */
__device__
fcm_ref_t
getFCM(uint level, uint partition, uint delay0)
{
	if(level == 0) {
		return tex2D(tf0_refs, (float) delay0, (float) partition);
	} else {
		return tex2D(tf1_refs, (float) delay0, (float) partition);
	}
}



__host__
cudaArray*
copyTable(size_t width,
		size_t height,
		const std::vector<fcm_ref_t>& h_table,
		const cudaChannelFormatDesc* channelDesc)
{
	assert(h_table.size() == width * height);

	cudaArray* d_table;
	CUDA_SAFE_CALL(cudaMallocArray(&d_table, channelDesc, width, height));

	size_t bytes = height * width * sizeof(fcm_ref_t);
	CUDA_SAFE_CALL(cudaMemcpyToArray(d_table, 0, 0, &h_table[0], bytes, cudaMemcpyHostToDevice));
	return d_table;
}



__host__
cudaArray*
f0_setDispatchTable(
		size_t partitionCount,
		size_t delayCount,
		const std::vector<fcm_ref_t>& h_table)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<fcm_ref_t>();
	cudaArray* d_table =
		copyTable(delayCount, partitionCount, h_table, &channelDesc);
	// set texture parameters
	tf0_refs.addressMode[0] = cudaAddressModeClamp;
	tf0_refs.addressMode[1] = cudaAddressModeClamp;
	tf0_refs.filterMode = cudaFilterModePoint;
	tf0_refs.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(tf0_refs, d_table, channelDesc));
	return d_table;
}



__host__
cudaArray*
f1_setDispatchTable(
		size_t partitionCount,
		size_t delayCount,
		const std::vector<fcm_ref_t>& h_table)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<fcm_ref_t>();
	cudaArray* d_table =
		copyTable(delayCount, partitionCount, h_table, &channelDesc);
	// set texture parameters
	tf1_refs.addressMode[0] = cudaAddressModeClamp;
	tf1_refs.addressMode[1] = cudaAddressModeClamp;
	tf1_refs.filterMode = cudaFilterModePoint;
	tf1_refs.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(tf1_refs, d_table, channelDesc));
	return d_table;
}



__host__
cudaArray*
f1_setDispatchTable2(
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
	tf1_refs2.addressMode[0] = cudaAddressModeClamp;
	tf1_refs2.addressMode[1] = cudaAddressModeClamp;
	tf1_refs2.addressMode[2] = cudaAddressModeClamp;
	tf1_refs2.filterMode = cudaFilterModePoint;
	tf1_refs2.normalized = false;
	CUDA_SAFE_CALL(cudaBindTextureToArray(tf1_refs2, d_table, channelDesc));
	return d_table;
}

/* At run-time we can load the relevant part of the dispatch table from texture
 * memory to shared memory. Both the shared memory arrays here should be of
 * length MAX_DELAY */
//! \todo make level a template parameter
__device__
void
loadDispatchTable_(uint level, uint32_t* s_fcmAddr[], ushort2 s_fcmPitch[])
{
	if(threadIdx.x < MAX_DELAY) {
		fcm_ref_t fcm = getFCM(level, CURRENT_PARTITION, threadIdx.x);
		s_fcmAddr[threadIdx.x] = f0_base(fcm);
		s_fcmPitch[threadIdx.x].x = f0_pitch(fcm);
		s_fcmPitch[threadIdx.x].y = DIV_CEIL(f0_pitch(fcm), THREADS_PER_BLOCK);
	}
	__syncthreads();
}

#endif
