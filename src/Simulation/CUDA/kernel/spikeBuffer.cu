//! \file spikeBuffer.cu

/* For L1 delivery spikes are delivered via global memory. To reduce the number
 * of non-coalesced global memory accesses, we first stage outgoing spikes in
 * shared-memory buffers. */

/* Each buffer contains one warp's worth of data, for efficient bandwidth
 * utilisation when flushing */
#define BUFFER_SZ 16

/*! At the very least one buffer is made available for each potential target
 * partition. To load-balance, multiple buffers may be allocated to the same
 * target partition, especially if the number of partitions is small 
 *
 * Max partition size used here because this is the size of the buffer as
 * allocated in kernel.cu:step. This is very brittle and will probably break.
 *
 * \todo use a c++ template to make this a compile time constant that depends
 * on total buffer size. */
#define MAX_BUFFER_COUNT (MAX_PARTITION_SIZE/(BUFFER_SZ*2)) // 2 since we use uint2


/* We can flush multiple buffers per block for better global memory bandwidth
 * utilisation */
#define THREADS_PER_BUFFER (BUFFER_SZ*2)

#define BUFFERS_PER_BLOCK (THREADS_PER_BLOCK / THREADS_PER_BUFFER)


//! \todo move to separate file for global buffer
/*! \return
 *      word offset to beginning of global memory spike buffer for a
 *      particular partition pair */
__device__
size_t
g_sbBase(size_t pitch, size_t src, size_t tgt, size_t bufferIdx)
{
	ASSERT(src < PARTITION_COUNT);
	ASSERT(tgt < PARTITION_COUNT);
	ASSERT(bufferIdx <= 1);
	return ((tgt * PARTITION_COUNT + src) * 2 + bufferIdx) * pitch;
}



__device__
uint
s_sbBuffersPerPartition()
{
	return MAX_BUFFER_COUNT / PARTITION_COUNT;
}



/* Due to rounding some buffers at the end may be invalid */
__device__
uint
s_sbCount()
{
	return s_sbBuffersPerPartition() * PARTITION_COUNT;
}




/*! \todo modify L1 delivery to handle more partitions than buffers. The L1 CM
 * needs to be split to do this. */


__device__
uint
s_sbBufferIdx(uint targetPartition, uint buffersPerPartition)
{
    ASSERT(targetPartition < PARTITION_COUNT);
    uint idx
        = targetPartition * buffersPerPartition
        + threadIdx.x % buffersPerPartition;
    ASSERT(idx < s_sbCount());
    return idx;
}


__device__
uint
s_sbPartition(uint buffer, uint buffersPerPartition)
{
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
g_headOffset(size_t src, size_t tgt, size_t pitch, size_t bufferIdx)
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
		size_t offset = g_headOffset(sourcePartition, CURRENT_PARTITION, pitch, bufferIdx);
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
 * After call s_heads contains number of entries filled in the global memory
 * buffer, so far.
 */
__device__
void
s_sbFlush(
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
	uint* s_outbuf32 = (uint*) s_outbuf64;
	uint* g_sb32 = (uint*) g_sb64;

	__shared__ uint g_offset[BUFFERS_PER_BLOCK]; /* ... into global buffer */

	uint t_buf = threadIdx.x / THREADS_PER_BUFFER;
	uint t_offset = threadIdx.x % THREADS_PER_BUFFER;

	uint targetPartition = s_sbPartition(outbufferIdx, buffersPerPartition);

	/* Several output buffers may write to the same global buffer, so we need
	 * atomic update. After this g_offset should contain the per shared memory
	 * buffer offsets used for writing to the per-partition global memory
	 * buffers  */
	if(t_offset == 0 && validBuffer) {
		g_offset[t_buf] = atomicAdd(s_heads + targetPartition, count);
	}
	__syncthreads();

	if(validBuffer && t_offset < count*2) {

		size_t base = g_sbBase(sbPitch, CURRENT_PARTITION, targetPartition, writeBufferIdx);
		uint data = s_outbuf32[outbufferIdx * BUFFER_SZ * 2 + t_offset];

		g_sb32[2 * (base + g_offset[t_buf]) + t_offset] = data;
#ifdef __DEVICE_EMULATION__
		if(threadIdx.x % 2 == 1) {
			DEBUG_MSG("Sending L1 current %f for synapse %d-?? -> %u-%u"
					"(after unknown delay)\n",
					__int_as_float(s_outbuf32[outbufferIdx * BUFFER_SZ * 2 + t_offset + 1]),
					CURRENT_PARTITION,
					targetPartition,
					targetNeuron(s_outbuf32[outbufferIdx * BUFFER_SZ * 2 + t_offset]));
		}
#endif
	}
	__syncthreads();
}


/*! Flush all shared memory spike buffers to global memory */
__device__
void
s_sbFlushAll(
	uint buffersPerPartition,
	uint writeBufferIdx,
	size_t headPitch,
	uint32_t* g_heads,
	uint32_t* s_heads,
	uint* s_outheads,
	uint2* s_sb,
	uint2* g_sb,
	size_t sbPitch)
{
	uint bcount = s_sbCount();

	/* Flush all buffers which still have data in them */
	for(uint flush_i=0; flush_i < bcount; flush_i += BUFFERS_PER_BLOCK) {
		uint bufferIdx = flush_i + threadIdx.x / THREADS_PER_BUFFER;
		s_sbFlush(
			bufferIdx < bcount,
			bufferIdx,
			writeBufferIdx,
			buffersPerPartition,
			s_outheads[bufferIdx],
			s_heads,
			s_sb,
			g_sb, sbPitch);
	}

	/* Now that all buffers have been flushed, s_heads should contain the total
	 * number of spikes written to the global buffer. This needs to be written
	 * to the global memory head buffer so that the target partition knows how
	 * many spikes are valid */
	//! \todo factor out function
	__syncthreads(); /* ensure s_heads is up to date */

	uint targetPartition = threadIdx.x;
	ASSERT(MAX_PARTITION_COUNT <= THREADS_PER_BLOCK);
	if(targetPartition < PARTITION_COUNT) {
		size_t g_offset = g_headOffset(CURRENT_PARTITION, targetPartition, headPitch, writeBufferIdx);
		//! \todo only write if the buffer contains anything
		DEBUG_MSG("Buffer %u->%u (outgoing) contains %u spikes\n",
				CURRENT_PARTITION, targetPartition, s_heads[targetPartition]);
		g_heads[g_offset] = s_heads[targetPartition];
	}
}
