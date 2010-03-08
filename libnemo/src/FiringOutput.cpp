/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "FiringOutput.hpp"
#include "util.h"
#include "bitvector.cu_h"
#include "nemo_types.h"

namespace nemo {

FiringOutput::FiringOutput(
		size_t partitionCount,
		size_t partitionSize,
		uint maxReadPeriod):
	md_buffer(NULL),
	mh_buffer(NULL),
	m_pitch(0),
	m_partitionCount(partitionCount),
	m_bufferedCycles(0),
    m_maxReadPeriod(maxReadPeriod),
	md_allocated(0),
	m_partitionSize(partitionSize)
{
	size_t width = BV_BYTE_PITCH;
	size_t height = partitionCount * maxReadPeriod;

	size_t bytePitch;
	CUDA_SAFE_CALL(cudaMallocPitch((void**)(&md_buffer), &bytePitch, width, height));
	CUDA_SAFE_CALL(cudaMemset2D(md_buffer, bytePitch, 0, bytePitch, height));
	m_pitch = bytePitch / sizeof(uint32_t);

	size_t md_allocated = bytePitch * height;
	CUDA_SAFE_CALL(cudaMallocHost((void**) &mh_buffer, md_allocated));
	memset(mh_buffer, 0, md_allocated);
}



FiringOutput::~FiringOutput()
{
	cudaFreeHost(mh_buffer);
	cudaFree(md_buffer);
}



uint32_t*
FiringOutput::step()
{
	uint32_t* ret = md_buffer + m_bufferedCycles * m_partitionCount * m_pitch;
	m_bufferedCycles += 1;
	if(m_bufferedCycles > m_maxReadPeriod) {
		fprintf(stderr, "WARNING: firing buffer overflow. Firing buffer flushed\n");
		m_bufferedCycles = 0;
		ret = md_buffer;
	}
	return ret;
}



uint
FiringOutput::readFiring(
		const std::vector<uint>** cycles,
		const std::vector<uint>** neuronIdx)
{
	m_cycles.clear();
	m_neuronIdx.clear();

	//! \todo error handling
	CUDA_SAFE_CALL(cudaMemcpy(mh_buffer,
				md_buffer,
				m_bufferedCycles * m_partitionCount * m_pitch * sizeof(uint32_t),
				cudaMemcpyDeviceToHost));
	populateSparse(m_bufferedCycles, mh_buffer, m_cycles, m_neuronIdx);

	*cycles = &m_cycles;
	*neuronIdx = &m_neuronIdx;
	uint readCycles = m_bufferedCycles;
	m_bufferedCycles = 0;
	return readCycles;
}



void
FiringOutput::populateSparse(
		uint bufferedCycles,
		const uint32_t* hostBuffer,
		std::vector<uint>& firingCycle,
		std::vector<uint>& neuronIdx)
{
	for(uint cycle=0; cycle < bufferedCycles; ++cycle) {
		size_t cycleOffset = cycle * m_partitionCount * m_pitch;

		for(size_t partition=0; partition < m_partitionCount; ++partition) {
			size_t partitionOffset = cycleOffset + partition * m_pitch;

			for(size_t nword=0; nword < m_pitch; ++nword) {

				/* Within a partition we might go into the padding part of the
				 * firing buffer. We rely on the device not leaving any garbage
				 * in the unused entries */
				uint32_t word = hostBuffer[partitionOffset + nword];

				//! \todo skip loop if nothing is set
				for(size_t nbit=0; nbit < 32; ++nbit) {

					bool fired = word & (1 << nbit);
					if(fired) {
						nidx_t nidx = partition * m_partitionSize + nword*32 + nbit;
						firingCycle.push_back(cycle);	
						neuronIdx.push_back(nidx);
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

} // end namespace nemo
