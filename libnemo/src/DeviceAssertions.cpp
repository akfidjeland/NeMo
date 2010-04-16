/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "DeviceAssertions.hpp"

#include <sstream>
#include <cuda_runtime.h>

#include "device_assert.cu_h"
#include "kernel.cu_h"


DeviceAssertions::DeviceAssertions(unsigned partitionCount) :
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
DeviceAssertions::check(unsigned cycle) throw (DeviceAssertionFailure)
{
#ifdef DEVICE_ASSERTIONS
	int* h_mem = &mh_mem[0];
	::getDeviceAssertions(m_partitionCount, h_mem);

	for(unsigned partition=0; partition < m_partitionCount; ++partition) {
		for(unsigned thread=0; thread < THREADS_PER_BLOCK; ++thread) {
			int line = h_mem[assertion_offset(partition, thread)];
			if(line != 0) {
				throw DeviceAssertionFailure(partition, thread, line, cycle);
			}
		}
	}
#endif
	return;
}



DeviceAssertionFailure::DeviceAssertionFailure(unsigned partition,
		unsigned thread, unsigned line, unsigned cycle)
{
	std::ostringstream msg;
	msg << "Device assertion failure for partition "
		<< partition << " thread " << thread << " in line "
		<< line << " during cycle " << cycle
		<< ". Only the first assertion failure is reported and the exact file is not known"
		<< std::endl;
	m_what = msg.str();

}
