#ifndef NEMO_CUDA_EXCEPT_HPP
#define NEMO_CUDA_EXCEPT_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <sstream>

#include <cuda_runtime.h>
#include <boost/format.hpp>

#include <nemo/exception.hpp>
#include <nemo/nemo_error.h>

namespace nemo {
	namespace cuda {

using boost::format;

class DeviceAllocationException : public nemo::exception
{
	public :

		DeviceAllocationException(const char* structname,
				size_t bytes,
				cudaError err) :
			nemo::exception(NEMO_CUDA_MEMORY_ERROR,
					str(format("Failed to allocate %uB for %s.\nCuda error: %s\n")
						% bytes % structname % cudaGetErrorString(err)))
		{}
};


class KernelInvocationError : public nemo::exception
{
	public :
		KernelInvocationError(cudaError_t status) :
			nemo::exception(
					NEMO_CUDA_INVOCATION_ERROR,
					cudaGetErrorString(status)) {}
};

	} // end namespace cuda
} // end namespace nemo


#define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                  \
    if(cudaSuccess != err) {                                               \
        std::ostringstream msg;                                            \
        msg << "Cuda error in file " << __FILE__ << " in line "            \
            << __LINE__ << ": " << cudaGetErrorString(err);                \
        throw nemo::exception(NEMO_CUDA_INVOCATION_ERROR, msg.str().c_str());\
    } }

#endif
