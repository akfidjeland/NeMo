/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda_runtime.h>

#include "device_memory.hpp"
//! \todo merge exception into this class
#include "exception.hpp"

namespace nemo {
	namespace cuda {


void
safeCall(cudaError_t err, unsigned error = NEMO_CUDA_MEMORY_ERROR)
{
	if(cudaSuccess != err) {
		throw nemo::exception(error, cudaGetErrorString(err));
	}
}


void
d_malloc(void** d_ptr, size_t sz, const char* name)
{
	cudaError_t err = cudaMalloc(d_ptr, sz);
	if(cudaSuccess != err) {
		throw DeviceAllocationException(name, sz, err);
	}
}



void
d_free(void* arr)
{
	safeCall(cudaFree(arr));
}



void
d_mallocPitch(void** d_ptr, size_t* bpitch, size_t width, size_t height, const char* name)
{
	if(width == 0) {
		*d_ptr = NULL;
		*bpitch = 0;
		return;
	}
	cudaError_t err = cudaMallocPitch(d_ptr, bpitch, width, height);
	if(cudaSuccess != err) {
		/* We throw a special exception here, as we catch these internally to
		 * report on current memory usage */
		throw DeviceAllocationException(name, height * width, err);
	}
}



void
memcpyToDevice(void* dst, const void* src, size_t count)
{
	safeCall(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}


void
memcpyFromDevice(void* dst, const void* src, size_t count)
{
	safeCall(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}


void
d_memset2D(void* d_ptr, size_t pitch, int value, size_t height)
{
	safeCall(cudaMemset2D(d_ptr, pitch, value, pitch, height));
}


void
mallocPinned(void** h_ptr, size_t sz)
{
	safeCall(cudaMallocHost(h_ptr, sz));
}


void
freePinned(void* arr)
{
	safeCall(cudaFreeHost(arr));
}


} 	} // end namespaces
