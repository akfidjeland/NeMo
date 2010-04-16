#ifndef INCOMING_CU
#define INCOMING_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "incoming.cu_h"


__constant__ size_t c_incomingPitch; // word pitch


__host__
void
setIncomingPitch(size_t pitch)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_incomingPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice));
}



/*! \return the buffer number to use for the given delay, given current cycle */
__device__
unsigned
incomingSlot(unsigned cycle, unsigned delay1)
{
	return (cycle + delay1) % MAX_DELAY;
}



/* Return offset into full buffer data structure to beginning of buffer for a
 * particular targetPartition and a particular delay. */
__device__
unsigned
incomingBufferStart(unsigned targetPartition, unsigned cycle, unsigned delay1)
{
	return (targetPartition * MAX_DELAY + incomingSlot(cycle, delay1)) * c_incomingPitch;
}



__device__
incoming_t
getIncoming(unsigned cycle, unsigned offset, incoming_t* g_incoming)
{
	return g_incoming[incomingBufferStart(CURRENT_PARTITION, cycle, 0) + offset];
}



/*! \return incoming spike group from a particular source */
__device__ incoming_t make_incoming(unsigned warpOffset) { return warpOffset; }

__device__ unsigned incomingWarpOffset(incoming_t in) { return in; }


/*! \return address into matrix with number of incoming synapse groups */
__device__
size_t
incomingCountAddr(unsigned targetPartition, unsigned cycle, unsigned delay1)
{
	return targetPartition * MAX_DELAY + incomingSlot(cycle, delay1);
}

#endif
