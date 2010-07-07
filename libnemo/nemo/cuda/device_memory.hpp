#ifndef NEMO_CUDA_DEVICE_MEMORY_H
#define NEMO_CUDA_DEVICE_MEMORY_H

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/* Wrapper for CUDA device memory methods, with appropriate error handling.
 * This wrapper also helps to reduce cuda_runtime header dependencies. */

namespace nemo {
	namespace cuda {

/*! Allocate 2D memory on the device
 *
 * \param bytePitch actual byte pitch after allignment
 * \param width desired width in bytes
 * \param name for debugging purposes
 */
void
d_mallocPitch(void** d_ptr, size_t* bytePitch, size_t width, size_t height, const char* name);


void
mallocPinned(void** h_ptr, size_t sz);

void
freePinned(void* arr);

} 	} // end namespaces

#endif
