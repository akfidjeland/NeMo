#ifndef EXCEPT_HPP
#define EXCEPT_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

class DeviceAllocationException : public std::exception
{
	public :

		DeviceAllocationException(const char* structname,
				size_t bytes,
				cudaError err)
		{
			std::ostringstream msg;
			msg << "Failed to allocate " << bytes
				<< " B for " << structname << std::endl
				<< "CUDA error: " << cudaGetErrorString(err) << std::endl;
			m_msg = msg.str();
		}

		~DeviceAllocationException() throw () {}

		const char* what() const throw () { return m_msg.c_str(); }

	private :

		std::string m_msg;
};


class KernelInvocationError : public std::runtime_error
{
	public :

		KernelInvocationError(cudaError_t status) :
			std::runtime_error(cudaGetErrorString(status)) {}
};


#define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                  \
    if(cudaSuccess != err) {                                               \
        std::ostringstream msg;                                            \
        msg << "Cuda error in file " << __FILE__ << " in line "            \
            << __LINE__ << ": " << cudaGetErrorString(err);                \
        throw std::runtime_error(msg.str().c_str());                       \
    } }

#endif
