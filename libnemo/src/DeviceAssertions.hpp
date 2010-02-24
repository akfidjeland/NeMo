#ifndef DEVICE_ASSERTIONS_HPP
#define DEVICE_ASSERTIONS_HPP

/*! \brief Run-time assertions on the GPU
 *
 * If the kernel is compiled with device assertions (CPP flag
 * DEVICE_ASSERTIONS), the kernel can perform run-time assertions, logging
 * location data to global memory. Only the line-number is recorded, so some
 * guess-work my be required to work out exactly what assertion failed. There
 * is only one assertion failure slot per thread, so it's possible to overwrite
 * an assertion failure.
 *
 * \author Andreas Fidjeland
 */

#include <vector>
#include <stdexcept>
#include <string>


class DeviceAssertionFailure : public std::exception
{
	public :

		DeviceAssertionFailure(uint partition, uint thread, uint line, uint cycle);

		~DeviceAssertionFailure() throw () {}

		const char* what() const throw() { return m_what.c_str(); }

	private :

		std::string m_what;
};



class DeviceAssertions
{
	public :

		/* Allocate device memory for device assertions */
		DeviceAssertions(uint partitionCount);

		/* Check whether any device assertions have failed. Only the last
		 * assertion failure for each thread will be reported. Checking device
		 * assertions require reading device memory and can therefore be
		 * costly. */
		void check(uint cycle) throw(DeviceAssertionFailure);

	private :

		uint m_partitionCount;

		std::vector<int> mh_mem;
};

#endif
