/* For L1 delivery spikes are delivered via global memory. To reduce the number
 * of non-coalesced global memory accesses, we first stage outgoing buffers in
 * shared memory.
 *
 * At the very least one buffer is made available for each potential target
 * partition. To load-balance, multiple buffers may be allocated to the same
 * target partition, especially if the number of partitions is small */




/*! \return
 *      word offset to beginning of global memory spike buffer for a particular
 *      partition pair */
__device__
size_t
sbBase(size_t pitch, size_t src, size_t tgt, size_t bufferIdx)
{
	ASSERT(src < PARTITION_COUNT);
	ASSERT(tgt < PARTITION_COUNT);
	ASSERT(bufferIdx <= 1);
	return ((tgt * PARTITION_COUNT + src) * 2 + bufferIdx) * pitch;
}


#define BUFFER_SZ 16
/*! Max partition size used because this is the size of the buffer as allocated
 * in kernel.cu:step. This is very brittle and will probably break.
 *
 * \todo use a c++ template to make this a compile time constant that depends
 * on total buffer size. */
#define MAX_BUFFER_COUNT (MAX_PARTITION_SIZE/(BUFFER_SZ*2)) // 2 since we use uint2

/* We can flush multiple buffers per block for better global memory bandwidth
 * utilisation */
#define THREADS_PER_BUFFER (BUFFER_SZ*2)
#define BUFFERS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_BUFFER)


__device__
uint
buffersPerPartition()
{
	return MAX_BUFFER_COUNT / PARTITION_COUNT;
}



/* Due to rounding some buffers at the end may be invalid */
__device__
uint
bufferCount()
{
	return buffersPerPartition() * PARTITION_COUNT;
}




/*! \todo modify L1 delivery to handle more partitions than buffers. The L1 CM
 * needs to be split to do this. */


__device__
uint
outputBufferIdx(uint targetPartition, uint buffersPerPartition)
{
    ASSERT(targetPartition < PARTITION_COUNT);
    uint idx
        = targetPartition * buffersPerPartition
        + threadIdx.x % buffersPerPartition;
    ASSERT(idx < bufferCount());
    return idx;
}


__device__
uint
bufferPartition(uint buffer, uint buffersPerPartition)
{
	//! \todo tidy!
    //uint buffersPerPartition = MAX_BUFFER_COUNT / PARTITION_COUNT;
    return buffer / buffersPerPartition;
}






//! \todo generalise this function, to have static size and type
//! \todo move this somewhere else
template<int SIZE>
__device__
void
s_clear(uint* s_buf)
{
    /* Loop should be unrolled by compiler in most cases */
    for(uint i=0; i < DIV_CEIL(SIZE, THREADS_PER_BLOCK); ++i) {
        size_t idx = i*THREADS_PER_BLOCK + threadIdx.x;
        if(idx < SIZE) {
            s_buf[idx] = 0;
        }
    }
}


template<int SIZE>
__device__
void
s_clear_(uint* s_buf)
{
    s_clear<SIZE>(s_buf);
    __syncthreads();
}



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
 * particular partition pair
 * 
 * \param src Source partition
 * \param tgt Target partition
 * \param pitch
 *		Pitch of global memory spike buffer
 * \param bufferIdx
 *		Index of double buffer (0 or 1)
 */
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
 * \param count
 *		Number of entries from the output buffer to flush
 *
 * \todo write 4B values instead of 8B
 */
__device__
void
flushSpikeBuffer(
	bool validBuffer,
	uint outbufferIdx,
	uint writeBufferIdx,
	uint buffersPerPartition,
	uint count,
	uint* s_heads,
	uint2* s_outbuf64,
	uint2* g_sb64,
	size_t sbPitch)
{
	/* We only have one warp's worth of data here. To improve bandwidth
	 * utilisation write 4B per thread rather than 8B. */
	//uint* s_outbuf32 = (uint*) s_outbuf64;
	//uint* g_sb32 = (uint*) g_sb64;

	__shared__ uint g_offset[BUFFERS_PER_BLOCK]; /* ... into global buffer */

	uint t_buf = threadIdx.x / THREADS_PER_BUFFER;
	uint t_offset = threadIdx.x % THREADS_PER_BUFFER;
	bool validEntry = validBuffer && t_offset < count;

	uint targetPartition = bufferPartition(outbufferIdx, buffersPerPartition);

	/* Several output buffers may write to the same global buffer, so we
	 * need atomic update. */
	if(t_offset == 0 && validEntry) {
		g_offset[t_buf] = atomicAdd(s_heads + targetPartition, count);
	}
	__syncthreads();

	if(validEntry) {

		size_t base = sbBase(sbPitch, CURRENT_PARTITION, targetPartition, writeBufferIdx);
		//uint data = s_outbuf32[outbufferIdx * BUFFER_SZ * 2 + t_offset];
		uint2 data = s_outbuf64[outbufferIdx * BUFFER_SZ + t_offset];

		//g_sb32[2 * (base + g_offset[t_buf]) + t_offset] = data;
		g_sb64[base + g_offset[t_buf] + t_offset] = data;
		DEBUG_MSG("Sending L1 current %f for synapse %d-?? -> %u-%u"
				"(after unknown delay)\n",
				__int_as_float(data.y), CURRENT_PARTITION,
				targetPartition, targetNeuron(data.x));
	}
}


/*! Flush all spike buffers */
__device__
void
flushAllSpikeBuffers(
	uint buffersPerPartition,
	uint writeBufferIdx,
	size_t headPitch,
	uint32_t* g_heads,
	uint32_t* s_heads,
	uint* s_outheads,
	uint2* s_outbuf,
	uint2* g_sq,
	size_t sbPitch)
{
	uint bcount = bufferCount();

	/* Flush all buffers which still have data in them */
	for(uint flush_i=0; flush_i < bcount; flush_i += BUFFERS_PER_BLOCK) {
		uint bufferIdx = flush_i + threadIdx.x / THREADS_PER_BUFFER;
		flushSpikeBuffer(
			bufferIdx < bcount,
			bufferIdx,
			writeBufferIdx,
			buffersPerPartition,
			s_outheads[bufferIdx],
			s_heads,
			s_outbuf,
			g_sq, sbPitch);
	}

	/* Now that all buffers have been flushed, s_heads should contain the total
	 * number of spikes written to the global buffer. This needs to be written
	 * to the global memory head buffer so that the target partition knows how
	 * many spikes are valid */
	//! \todo factor out function
	__syncthreads(); /* ensure s_heads is up to date */
	uint bufferIdx = threadIdx.x;
	ASSERT(MAX_BUFFER_COUNT <= THREADS_PER_BLOCK);
	if(bufferIdx < bcount) {
		uint targetPartition = bufferPartition(bufferIdx, buffersPerPartition);
		size_t g_offset = headOffset(CURRENT_PARTITION, targetPartition, headPitch, writeBufferIdx);
		//! \todo only write if the buffer contains anything
		g_heads[g_offset] = s_heads[targetPartition];
	}
}
