#ifndef PARTITION_CONFIGURATION_CU
#define PARTITION_CONFIGURATION_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"

/* Per-partition configuration */

__constant__ unsigned c_partitionSize[MAX_THREAD_BLOCKS];

__host__
cudaError
configurePartitionSize(const unsigned* d_partitionSize, size_t len)
{
	//! \todo set padding to zero
	assert(len <= MAX_THREAD_BLOCKS);
	return cudaMemcpyToSymbol(
			c_partitionSize,
			(void*) d_partitionSize,
			MAX_THREAD_BLOCKS*sizeof(unsigned),
			0, cudaMemcpyHostToDevice);
}

#endif
