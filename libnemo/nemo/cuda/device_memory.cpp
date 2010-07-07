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


} 	} // end namespaces
