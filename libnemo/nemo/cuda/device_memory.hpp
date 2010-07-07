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

#include <vector>
#include <nemo/exception.hpp>

namespace nemo {
	namespace cuda {


void
d_malloc(void** d_ptr, size_t sz, const char* name);


void
d_free(void*);


/*! Allocate 2D memory on the device
 *
 * \param bytePitch actual byte pitch after allignment
 * \param width desired width in bytes
 * \param name for debugging purposes
 */
void
d_mallocPitch(void** d_ptr, size_t* bytePitch, size_t width, size_t height, const char* name);


void
memcpyToDevice(void* dst, const void* src, size_t count);


/*
 * \param count
 * 		Number of elements to copy from vector
 *
 * \pre
 * 		size < vec.size()
 */
template<typename T>
void
memcpyToDevice(void* dst, const std::vector<T>& vec, size_t count)
{
	if(vec.empty()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "cannot copy empty vector to device");
	}
	if(count < vec.size()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "cannot copy more than length of vector");
	}
	memcpyToDevice(dst, &vec[0], count * sizeof(T));
}


template<typename T>
void
memcpyToDevice(void* dst, const std::vector<T>& vec)
{
	memcpyToDevice(dst, vec, vec.size());
}


void
memcpyFromDevice(void* dst, const void* src, size_t count);


/* \param count
 * 		Number of elements to copy from device into vector
 */
template<typename T>
void
memcpyFromDevice(std::vector<T>& vec, const void* src, size_t count)
{
	//! \todo perhaps we should do our own resizing here?
	if(vec.empty()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "attempt to copy into empty vector");
	}
	if(count > vec.size()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "attempt to copy into vector which is too small");
	}
	memcpyFromDevice(&vec[0], src, count * sizeof(T));
}

void
mallocPinned(void** h_ptr, size_t sz);

void
freePinned(void* arr);

} 	} // end namespaces

#endif
