#ifndef NEMO_CUDA_DEVICE_MEMORY_HPP
#define NEMO_CUDA_DEVICE_MEMORY_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file device_memory.hpp Device memory methods
 *
 * These are wrappers for CUDA device memory methods, with appropriate error
 * handling. This wrapper also serves to reduce cuda_runtime header
 * dependencies.
 */

#include <vector>
#include <boost/shared_array.hpp>
#include <nemo/exception.hpp>

namespace nemo {
	namespace cuda {


void
d_malloc(void** d_ptr, size_t sz, const char* name);


void
d_free(void*);


/*! Allocate memory block and put in smart pointer
 *
 * \param len length in /words/
 * \param name name of data structure (for error reporting)
 */
template<typename T>
boost::shared_array<T>
d_array(size_t len, const char* name)
{
	void* d_ptr = NULL;
	d_malloc(&d_ptr, len * sizeof(T), name);
	return boost::shared_array<T>(static_cast<T*>(d_ptr), d_free);
}


/*! Allocate 2D memory on the device
 *
 * \param[out] d_ptr device pointer to allocated memory
 * \param[in] bytePitch actual byte pitch after allignment
 * \param[in] width desired width in bytes
 * \param[in] height
 * \param[in] name of data structure (for error handling/debugging purposes)
 */
void
d_mallocPitch(void** d_ptr, size_t* bytePitch, size_t width, size_t height, const char* name);


void
memcpyBytesToDevice(void* dst, const void* src, size_t count);



template<typename T>
void
memcpyToDevice(T* dst, const T* src, size_t words)
{
	memcpyBytesToDevice((void*)dst, (void*)src, words * sizeof(T));
}


/*
 * \param count
 * 		Number of elements to copy from vector
 * \pre
 * 		count =< vec.size()
 */
template<typename T>
void
memcpyToDevice(T* dst, const std::vector<T>& vec, size_t words)
{
	if(vec.empty()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "cannot copy empty vector to device");
	}
	if(words > vec.size()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "cannot copy more than length of vector");
	}
	memcpyToDevice(dst, &vec[0], words);
}


template<typename T>
void
memcpyToDevice(T* dst, const std::vector<T>& vec)
{
	memcpyToDevice(dst, vec, vec.size());
}



void
memcpyBytesFromDevice(void* dst, const void* src, size_t bytes);



template<typename T>
void
memcpyFromDevice(T* dst, const T* src, size_t words)
{
	memcpyBytesFromDevice((void*)dst, (void*)src, words * sizeof(T));
}


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
	memcpyBytesFromDevice(&vec[0], src, count * sizeof(T));
}


void d_memset(void* d_ptr, int value, size_t count);

void d_memset2D(void* d_ptr, size_t bytePitch, int value, size_t height);


void
mallocPinned(void** h_ptr, size_t sz);

void
freePinned(void* arr);

} 	} // end namespaces

#endif
