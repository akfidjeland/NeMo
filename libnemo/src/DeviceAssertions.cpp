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
DeviceAssertions::check(uint cycle) throw (DeviceAssertionFailure)
{
#ifdef DEVICE_ASSERTIONS
	int* h_mem = &mh_mem[0];
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
		uint thread, uint line, uint cycle)
{
	std::ostringstream msg;
	msg << "Device assertion failure for partition "
		<< partition << " thread " << thread << " in line "
		<< line << " during cycle " << cycle
		<< ". Only the first assertion failure is reported and the exact file is not  known" << std::endl;
	m_what = msg.str();

}
