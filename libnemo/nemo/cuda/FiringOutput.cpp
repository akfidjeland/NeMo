/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>

#include "exception.hpp"
#include "FiringOutput.hpp"
#include "bitvector.cu_h"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {

FiringOutput::FiringOutput(
		const Mapper& mapper,
		unsigned maxReadPeriod):
	m_pitch(0),
	m_bufferedCycles(0),
	m_maxReadPeriod(maxReadPeriod),
	md_allocated(0),
	m_mapper(mapper)
{
	size_t width = BV_BYTE_PITCH;
	size_t height = m_mapper.partitionCount() * maxReadPeriod;

	size_t bytePitch;
	uint32_t* d_buffer;
	d_mallocPitch((void**)(&d_buffer), &bytePitch, width, height, "firing output");
	md_buffer = boost::shared_ptr<uint32_t>(d_buffer, d_free);
	d_memset2D(d_buffer, bytePitch, 0, height);
	m_pitch = bytePitch / sizeof(uint32_t);

	size_t md_allocated = bytePitch * height;
	uint32_t* h_buffer;

	mallocPinned((void**) &h_buffer, md_allocated);
	mh_buffer = boost::shared_ptr<uint32_t>(h_buffer, freePinned);
	memset(h_buffer, 0, md_allocated);
}



uint32_t*
FiringOutput::step()
{
	uint32_t* ret = md_buffer.get() + m_bufferedCycles * m_mapper.partitionCount() * m_pitch;
	m_bufferedCycles += 1;
	if(m_bufferedCycles > m_maxReadPeriod) {
		m_bufferedCycles = 0;
		ret = md_buffer.get();
		throw nemo::exception(NEMO_BUFFER_OVERFLOW, "Firing buffer overflow");
	}
	return ret;
}



unsigned
FiringOutput::readFiring(
		const std::vector<unsigned>** cycles,
		const std::vector<unsigned>** neuronIdx)
{
	m_cycles.clear();
	m_neuronIdx.clear();

	memcpyFromDevice(mh_buffer.get(), md_buffer.get(),
				m_bufferedCycles * m_mapper.partitionCount() * m_pitch * sizeof(uint32_t));
	populateSparse(m_bufferedCycles, mh_buffer.get(), m_cycles, m_neuronIdx);

	*cycles = &m_cycles;
	*neuronIdx = &m_neuronIdx;
	unsigned readCycles = m_bufferedCycles;
	m_bufferedCycles = 0;
	return readCycles;
}



void
FiringOutput::populateSparse(
		unsigned bufferedCycles,
		const uint32_t* hostBuffer,
		std::vector<unsigned>& firingCycle,
		std::vector<unsigned>& neuronIdx)
{
	unsigned pcount = m_mapper.partitionCount();
	for(unsigned cycle=0; cycle < bufferedCycles; ++cycle) {
		size_t cycleOffset = cycle * pcount * m_pitch;

		for(size_t partition=0; partition < pcount; ++partition) {
			size_t partitionOffset = cycleOffset + partition * m_pitch;

			for(size_t nword=0; nword < m_pitch; ++nword) {

				/* Within a partition we might go into the padding part of the
				 * firing buffer. We rely on the device not leaving any garbage
				 * in the unused entries */
				uint32_t word = hostBuffer[partitionOffset + nword];

				//! \todo skip loop if nothing is set
				for(size_t nbit=0; nbit < 32; ++nbit) {
					bool fired = (word & (1 << nbit)) != 0;
					if(fired) {
						firingCycle.push_back(cycle);	
						neuronIdx.push_back(m_mapper.hostIdx(partition, nword*32 + nbit));
					}
				}
			}
		}
	}
}



void
FiringOutput::flushBuffer()
{
	m_bufferedCycles = 0;
}

	} // end namespace cuda
} // end namespace nemo
