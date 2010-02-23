#include "DeviceAssertions.hpp"

#include <sstream>
#include <cuda_runtime.h>

#include "device_assert.cu_h"
#include "kernel.cu_h"


DeviceAssertions::DeviceAssertions(uint partitionCount) :
	m_partitionCount(partitionCount),
	mh_mem(partitionCount * THREADS_PER_BLOCK, 0)
{
	::clearDeviceAssertions();
	/* The amount of memory required to hold assertion data on the device is
	 * quite small, less than 100KB in the worst case. When used we need to
	 * copy from device to host every cycle. For these reasons we keep this in
	 * pinned memory */
	//! \todo use pinned memory here instead.
}



void
DeviceAssertions::check(uint cycle) throw (DeviceAssertionFailure, std::logic_error)
{
#ifdef DEVICE_ASSERTIONS
	int* h_mem = &mh_mem[0];
	if(h_mem == NULL) {
		throw std::logic_error("Device assertions checked without having been allocated");
	}

	::getDeviceAssertions(m_partitionCount, h_mem);

	for(uint partition=0; partition < m_partitionCount; ++partition) {
		for(uint thread=0; thread < THREADS_PER_BLOCK; ++thread) {
			int line = h_mem[assertion_offset(partition, thread)];
			if(line != 0) {
				throw DeviceAssertionFailure(partition, thread, line, cycle);
			}
		}
	}
#endif
	return;
}



DeviceAssertionFailure::DeviceAssertionFailure(uint partition,
		uint thread, uint line, uint cycle) :
	m_partition(partition),
	m_thread(thread),
	m_line(line),
	m_cycle(cycle)
{ }



const char*
DeviceAssertionFailure::what() const throw()
{
	std::ostringstream msg;
	msg << "Device assertion failure for partition "
		<< m_partition << " thread " << m_thread << " in line "
		<< m_line << " during cycle " << m_cycle
		<< ". Only the first assertion failure is reported and the exact file is not  known"; 
	return msg.str().c_str();	
}
