
/*! \return word offset to beginning of spike buffer for a particular partition
 * pair */
__device__
size_t
sbBase(size_t pitch, size_t src, size_t tgt, size_t bufferIdx)
{
	ASSERT(src < PARTITION_COUNT);
	ASSERT(tgt < PARTITION_COUNT);
	ASSERT(bufferIdx <= 1);
	return ((tgt * PARTITION_COUNT + src) * 2 + bufferIdx) * pitch;
}




//! \todo fix these constants

#define BUFFER_SZ 16
//! \todo modify L1 delivery to handle more buffers than threads
//#define BUFFER_COUNT MAX_PARTITION_COUNT
#define BUFFER_COUNT THREADS_PER_BLOCK



/*! Each possible source buffer has a counter (the 'buffer head') specifying
 * how many spikes are due for delivery. The buffer heads is a 2D matrix with
 * one row for each target partition
 *
 * The buffer head matrix is written to during spike scatter and read from
 * during spike gather. Since the accesses in these two cases are in different
 * order, so one access will necessarily be non-coalesced. It does not matter
 * which it is.
 *
 * \todo it would be possible to set the row pitch such that the non-coalesced
 * access avoids bank conflicts. It would still be non-coalesced, though.
 */


/*! \return the offset into the global memory buffer head matrix for a
 * particular partition pair */
__device__
size_t
headOffset(size_t src, size_t tgt, size_t pitch, size_t bufferIdx)
{
	ASSERT(src < PARTITION_COUNT);
	ASSERT(tgt < PARTITION_COUNT);
	ASSERT(bufferIdx <= 1);
    return bufferIdx * PARTITION_COUNT * pitch + tgt * pitch + src;
}



/*! Load and clear all the buffer heads when processing incoming spikes in the
 * target partition. The buffer heads need to be cleared so that the buffer is
 * ready to be filled again. Any synapses in the buffer is left there as
 * garbage. */
__device__
void
loadAndClearBufferHeads(
        uint32_t* g_heads,
        uint32_t* s_heads,
        size_t pitch,
        size_t bufferIdx)
{
#if MAX_THREAD_BLOCKS > THREADS_PER_BLOCK
#error	"Need to rewrite loadL1Current to load spike queue heads in several loads"
#endif
	int sourcePartition = threadIdx.x;
	if(sourcePartition < PARTITION_COUNT) {
		size_t offset = headOffset(sourcePartition, CURRENT_PARTITION, pitch, bufferIdx);
		//! \todo could use atomicExch here instead. Not sure which is faster.
		s_heads[sourcePartition] = g_heads[offset];
		g_heads[offset] = 0;
	}
	__syncthreads();
}




/* Flush spike buffer (up to maxSlot) for a single partition.
 *
 * \todo write 4B values instead of 8B
 */
__device__
void
flushSpikeBuffer(
	uint writeBufferIdx,
	uint count,
	int targetPartition,
	uint* s_heads,
	uint2* s_outbuf64,
	uint2* g_sq64,
	size_t sqPitch)
{
	/* We only have one warp's worth of data here. To improve bandwidth
	 * utilisation write 4B per thread rather than 8B. */
	// uint* s_outbuf32 = (uint*) s_outbuf64;
	// uint* g_sq32 = (uint*) g_sq64;

	if(threadIdx.x < count) {
		//! \todo simplify addressing once old L1CM is removed
		size_t base = sbBase(sqPitch, CURRENT_PARTITION, targetPartition, writeBufferIdx);
		//uint data = s_outbuf32[targetPartition * BUFFER_SZ * 2 + threadIdx.x];
		uint2 data = s_outbuf64[targetPartition * BUFFER_SZ + threadIdx.x];
		//g_sq32[2 * (base + s_heads[targetPartition]) + threadIdx.x] = data;
		g_sq64[base + s_heads[targetPartition] + threadIdx.x] = data;
		DEBUG_MSG("Sending L1 current %f for synapse %d-?? -> %u-%u (after unknown delay)\n",
			__int_as_float(data.y), CURRENT_PARTITION,
			targetPartition, targetNeuron(data.x));
	}
}


/*! Flush all spike buffers */
__device__
void
flushAllSpikeBuffers(
	uint writeBufferIdx,
	size_t headPitch,
	uint32_t* g_heads,
	uint32_t* s_heads,
	uint* s_outheads,
	uint2* s_outbuf,
	uint2* g_sq,
	size_t sqPitch)
{
	/* Determine global buffer offsets in parallel, and at the same time flush
	 * the buffer head to global memory. The head buffer is repurposed to now
	 * contain the offset into the buffer entry */
	uint targetPartition = threadIdx.x;
	if(targetPartition < PARTITION_COUNT) {
		size_t offset = headOffset(CURRENT_PARTITION, targetPartition, headPitch, writeBufferIdx);
		g_heads[offset] = s_heads[targetPartition] + s_outheads[targetPartition];
	}
	__syncthreads();

	//! \todo factor out function to flush all
	/* Now flush all buffers which still have data in them */
	for(int targetPartition=0; targetPartition<PARTITION_COUNT; ++targetPartition) {
		//! \todo could load the sizes in parallel here, without using atomics
		flushSpikeBuffer(
            writeBufferIdx,
			s_outheads[targetPartition],
			targetPartition,
			s_heads,
			s_outbuf,
			g_sq, sqPitch);
	}
}

