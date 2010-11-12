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
cudaError
setIncomingPitch(size_t pitch)
{
	return cudaMemcpyToSymbol(c_incomingPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}



/* Return offset into full buffer data structure to beginning of buffer for a
 * particular targetPartition and a particular delay. */
__device__
unsigned
incomingBufferStart(unsigned targetPartition, unsigned slot)
{
	//! \todo remove the MAX_DELAY factor here. Then inline into caller.
	return (targetPartition * MAX_DELAY + slot) * c_incomingPitch;
}




__device__
incoming_t
getIncoming(unsigned slot, unsigned offset, incoming_t* g_incoming)
{
	return g_incoming[incomingBufferStart(CURRENT_PARTITION, slot) + offset];
}



/*! \return incoming spike group from a particular source */
__device__ incoming_t make_incoming(unsigned warpOffset) { return warpOffset; }

__device__ unsigned incomingWarpOffset(incoming_t in) { return in; }



/*! \return address into matrix with number of incoming synapse groups
 * \param slot read or write slot
 *
 * \see readBuffer writeBuffer
 */
__device__
size_t
incomingCountAddr(unsigned targetPartition, unsigned slot)
{
	return targetPartition * MAX_DELAY + slot;
}


#endif
