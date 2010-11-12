#ifndef NEMO_CUDA_OUTGOING_CU
#define NEMO_CUDA_OUTGOING_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "outgoing.cu_h"

__constant__ size_t c_outgoingPitch;  // word pitch
__constant__ unsigned c_outgoingStep; // number of rows we can process in a thread block



__host__
outgoing_t
make_outgoing(pidx_t partition, unsigned warpOffset)
{
	assert(partition < MAX_PARTITION_COUNT);
	return make_uint2(partition, (unsigned) warpOffset);
}



__host__
cudaError
setOutgoingPitch(size_t targetPitch)
{
	return cudaMemcpyToSymbol(c_outgoingPitch, &targetPitch, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}


__host__
cudaError
setOutgoingStep(unsigned step)
{
	return cudaMemcpyToSymbol(c_outgoingStep, &step, sizeof(unsigned), 0, cudaMemcpyHostToDevice);
}


__host__ __device__
size_t
outgoingRow(pidx_t partition, nidx_t neuron, short delay0, size_t pitch)
{
	return outgoingCountOffset(partition, neuron, delay0) * pitch;
}



__device__ unsigned outgoingTargetPartition(outgoing_t out) { return out.x; } 
__device__ unsigned outgoingWarpOffset(outgoing_t out) { return out.y; }


__device__
outgoing_t
outgoing(unsigned presynaptic, unsigned delay0,
		unsigned jobIdx, outgoing_t* g_targets)
{
	size_t addr = outgoingRow(CURRENT_PARTITION, presynaptic, delay0, c_outgoingPitch);
	return g_targets[addr + jobIdx];
}




__host__ __device__
size_t
outgoingCountOffset(unsigned partition, short neuron, short delay0)
{
	//! \todo refactor after correctness is verified
	return partition * (MAX_PARTITION_SIZE * MAX_DELAY)
		+ neuron * MAX_DELAY
		+ delay0;
}


/*! \return
 *		the number of jobs for a particular firing neuron in the current
 *		partition */
__device__
unsigned
outgoingCount(short neuron, short delay0, unsigned* g_counts)
{
	return g_counts[outgoingCountOffset(CURRENT_PARTITION, neuron, delay0)];
}


#endif
