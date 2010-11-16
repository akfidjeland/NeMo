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
outgoing_addr_t
make_outgoing_addr(unsigned offset, unsigned len)
{
	return make_uint2(offset, len);
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
	return outgoingAddrOffset(partition, neuron, delay0) * pitch;
}



__device__ unsigned outgoingTargetPartition(outgoing_t out) { return out.x; } 
__device__ unsigned outgoingWarpOffset(outgoing_t out) { return out.y; }



/*! \return
 *		Address to the address info (!) for a particular neuron/delay pair
 */
__host__ __device__
size_t
outgoingAddrOffset(unsigned partition, short neuron, short delay0)
{
	return (partition * MAX_PARTITION_SIZE + neuron) * MAX_DELAY + delay0;
}



/*! \return
 *		The address info (offset/length) required to fetch the outgoing warp
 *		entries for a particular neuron/delay pair
 */
__device__
outgoing_addr_t
outgoingAddr(short neuron, short delay0, outgoing_addr_t* g_addr)
{
	return g_addr[outgoingAddrOffset(CURRENT_PARTITION, neuron, delay0)];
}

#endif
