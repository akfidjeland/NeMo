#include <cuda_runtime.h>
#include <cutil.h>
#include <string.h>

#include "FiringOutput.hpp"
#include "util.h"


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
	md_allocated(0)
{
	const size_t NEURONS_PER_BYTE = 8;
	size_t width = ALIGN(partitionSize, 32) / NEURONS_PER_BYTE;
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



void 
FiringOutput::readFiring(
		uint** cycles,
		uint** partitionIdx,
		uint** neuronIdx,
		uint* len,
		uint* totalCycles)
{
	m_cycles.clear();
	m_partitionIdx.clear();
	m_neuronIdx.clear();

	CUDA_SAFE_CALL(cudaMemcpy(mh_buffer,
				md_buffer,
				m_bufferedCycles * m_partitionCount * m_pitch * sizeof(uint32_t),
				cudaMemcpyDeviceToHost));
	populateSparse(m_bufferedCycles, mh_buffer,
			m_cycles, m_partitionIdx, m_neuronIdx);

	*len = m_cycles.size();
	*cycles = &m_cycles[0];
	*partitionIdx = &m_partitionIdx[0];
	*neuronIdx = &m_neuronIdx[0];
	*totalCycles = m_bufferedCycles;
	m_bufferedCycles = 0;
}



void
FiringOutput::populateSparse(
		uint bufferedCycles,
		const uint32_t* hostBuffer,
		std::vector<uint>& firingCycle,
		std::vector<uint>& partitionIdx,
		std::vector<uint>& neuronIdx)
{
	for(uint cycle=0; cycle < bufferedCycles; ++cycle) {
		size_t cycleOffset = cycle * m_partitionCount * m_pitch;

		for(size_t partition=0; partition < m_partitionCount; ++partition) {
			size_t partitionOffset = cycleOffset + partition * m_pitch;

			for(size_t nword=0; nword < m_pitch; ++nword) {

				/* Within a partition we might go into the padding part of the
				 * firing buffer.  We rely on the device not leaving any
				 * garbage in the unused entries */
				uint32_t word = hostBuffer[partitionOffset + nword];

				for(size_t nbit=0; nbit < 32; ++nbit) {

					bool fired = word & (1 << nbit);
					if(fired) {
						firingCycle.push_back(cycle);	
						partitionIdx.push_back(partition);
						neuronIdx.push_back(nword*32 + nbit);
					}
				}
			}
		}
	}
}
