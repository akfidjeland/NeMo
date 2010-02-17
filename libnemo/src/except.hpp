#ifndef EXCEPT_HPP
#define EXCEPT_HPP

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

#endif
