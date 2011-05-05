#ifndef NEMO_CUDA_RCM_CU
#define NEMO_CUDA_RCM_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "rcm.cu_h"


__host__ __device__
size_t
rcm_metaIndexAddress(pidx_t partition, nidx_t neuron)
{
	return partition * MAX_PARTITION_SIZE + neuron;
}



__device__
uint
rcm_indexRowStart(rcm_index_address_t addr)
{
	return addr.x;
}


__device__
uint
rcm_indexRowLength(rcm_index_address_t addr)
{
	return addr.y;
}



/*! \return address in RCM index for a neuron in current partition */
__device__
rcm_index_address_t
rcm_indexAddress(nidx_t neuron, const rcm_dt& rcm)
{
	return rcm.meta_index[rcm_metaIndexAddress(CURRENT_PARTITION, neuron)];
}


__device__
rcm_address_t
rcm_address(uint rowStart, uint rowOffset, const rcm_dt& rcm)
{
	return rcm.index[rowStart + rowOffset];
}



/*! \return word offset into RCM for a particular synapse */
__device__
size_t
rcm_offset(rcm_address_t warpOffset)
{
	return warpOffset * WARP_SIZE + threadIdx.x % WARP_SIZE;
}


#endif
