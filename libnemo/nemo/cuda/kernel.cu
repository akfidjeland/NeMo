//! \file kernel.cu

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <assert.h>

#include "kernel.cu_h"
#include "log.cu_h"


/*! Per-partition size
 *
 * Different partitions need not have exactly the same size. The exact size of
 * each partition is stored in constant memory, so that per-neuron loops can do
 * the correct minimum number of iterations
 */
__constant__ unsigned c_partitionSize[MAX_PARTITION_COUNT];


#include "device_assert.cu"
#include "bitvector.cu"
#include "double_buffer.cu"
#include "connectivityMatrix.cu"
#include "cycleCounting.cu"
#include "outgoing.cu"
#include "globalQueue.cu"
#include "nvector.cu"

#include "gather.cu"
#include "fire.cu"
#include "scatter.cu"
#include "stdp.cu"
#include "applySTDP.cu"


/*! Set partition size for each partition in constant memory
 * \see c_partitionSize */
__host__
cudaError
configurePartitionSize(const unsigned* d_partitionSize, size_t len)
{
	//! \todo set padding to zero
	assert(len <= MAX_PARTITION_COUNT);
	return cudaMemcpyToSymbol(
			c_partitionSize,
			(void*) d_partitionSize,
			MAX_PARTITION_COUNT*sizeof(unsigned),
			0, cudaMemcpyHostToDevice);
}
