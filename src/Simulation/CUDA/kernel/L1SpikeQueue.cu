#include "kernel.cu_h"
#include "util.h"
#include <stdint.h>
#ifdef __DEVICE_EMULATION__
#   include <assert.h>
#   include <stdio.h>
#endif


/* The current cycle indicates which half of the double buffer is for reading
 * and which is for writing */
__device__
uint
readBuffer(uint cycle)
{
    return (cycle & 0x1) ^ 0x1;
}


__device__
uint
writeBuffer(uint cycle)
{
    return cycle & 0x1;
}


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



/*! Update current buffer with incoming spikes */
__device__
void
updateCurrent(
		uint readBufferIdx, // double buffer index
		uint sourcePartition,
		uint32_t spikeIdx,   // index of spike for current thread
		uint32_t spikeCount, // number of spikes to delivered for current buffer 
		uint2* g_sq,
		size_t sqPitch,
		float* s_current)
{
#define BANKS 16
	//! \todo remove hard-coding
	//! \todo do this for all threads
	__shared__ uint s_committing[BANKS];
	if(threadIdx.x < BANKS) {
		s_committing[threadIdx.x] = 0;
	}
	__syncthreads();

	//! \todo conditional load from global memory

	uint targetPartition = CURRENT_PARTITION;
	size_t base = sbBase(sqPitch, sourcePartition, targetPartition, readBufferIdx);
	//! \todo load by 32-bit values
	uint2 spike = g_sq[base + spikeIdx];

	/* We don't need to clear the data from the spike buffer,
	 * as long as the head is cleared. \see loadAndClearBufferHeads */
	float weight = __int_as_float(spike.y);
	uint target = targetNeuron(spike.x);

	bool spiked = weight != 0.0f && spikeIdx < spikeCount;

	uint commitNo;
	if(spiked) {
		// serialise commits for each shared memory bank to avoid race condition
		commitNo = atomicAdd(s_committing + (target % BANKS), 1);
	}
	__syncthreads();

	/* In the worst case *every* spike has the same target. Determine the
	 * maximum number of threads that need to be serialised. */
	//! \todo use reduction to find the maximum here
	__shared__ uint s_maxCommit;
	if(threadIdx.x == 0) {
		s_maxCommit = 0;
		for(uint i=0; i<BANKS; ++i) {
			s_maxCommit = max(s_maxCommit, s_committing[i]);
		}
	}
	__syncthreads();

	for(uint commit=0; commit <= s_maxCommit; ++commit) {
		if(spiked && commitNo == commit) {
			s_current[targetNeuron(spike.x)] += weight;
			ASSERT(targetNeuron(spike.x) < MAX_PARTITION_SIZE);
			DEBUG_MSG("Receiving L1 current %f from %d-?? to %d-%d\n",
					weight, sourcePartition, targetPartition, targetNeuron(spike.x));
		}
		__syncthreads();
	}
	__syncthreads();
}



/*! Load all incoming spikes from L1 connetivity into current accumulator */
__device__
void
gatherL1Spikes_JIT_(
		uint readBufferIdx,
		uint2* g_sq,
		size_t sqPitch,
		uint* g_heads,
        size_t headPitch,
		float* s_current,
        uint32_t* s_heads)
{
	loadAndClearBufferHeads(g_heads, s_heads, headPitch, readBufferIdx);
	for(uint src=0; src<PARTITION_COUNT; ++src) {
		uint parallelLoads = DIV_CEIL(s_heads[src], THREADS_PER_BLOCK);
		for(uint load=0; load<parallelLoads; ++load) {
			uint spikeIdx = load * THREADS_PER_BLOCK + threadIdx.x;
			updateCurrent(readBufferIdx, src, spikeIdx, s_heads[src],
					g_sq, sqPitch, s_current);
		}
	}
	__syncthreads();
}


/* TODO: the loop structure here is nearly the same as deliverL0Spikes. Factor
 * out or use a code generator to avoid repetition */
__device__
void
deliverL1Spikes_JIT(
	uint maxDelay,
	uint writeBufferIdx,
	uint partitionSize,
	uint sf1_maxSynapses,
	uint* gf1_cm, uint f1_pitch, uint f1_size,
	uint32_t* s_recentFiring,
	//! \todo STDP support
#ifdef STDP
	//uint32_t* s_recentIncoming,
	//float* g_ltd,
	//uint stdpCycle,
#endif
	uint32_t* g_firingDelays,
	// L1 spike queue
    //! \todo allow more than 32 partitions (by splitting L1CM)
    uint2* s_outbuf,        // 16 words of buffer per target partition
	uint2* g_sq,
	size_t sqPitch,
	uint* g_heads,
    size_t headPitch)
{
	uint*  gf1_address =          gf1_cm + FCM_ADDRESS * f1_size;
	float* gf1_weights = (float*) gf1_cm + FCM_WEIGHT  * f1_size;

	/*! \note This is the maximum number of chunks required for this whole
	 * cluster. It should be possible to reduce this for rows with few entries.
	 * Perhaps better to just save the number of chunks in constant memory. It
	 * would depend on the chunk size, though. */
	__shared__ uint s_chunkCount;

	/* L1 spikes are delivered via a global memory buffer. Writes to these
	 * buffers may be quite scattered. To reduce the impact of non-coalesced
	 * writes we therefore stage synaptic data in shared memory before writing
	 * it to global memory.
	 *
	 * We therefore have two buffer heads to keep track of: one for the global
	 * per target-partition spike buffer (which may be filled from several
     * presynaptic neurons here), and one for the local per target-partition
     * buffer. The latter is the 'outbuffer', while the former is just
     * 'buffer'.  */

	/*! \todo use one of the general-purpose chunks of shared memory */
	/* Can't pack these into one array of uint2's, since we need to do atomic
	 * operations */
	__shared__ uint s_heads[BUFFER_COUNT];
	__shared__ uint s_outheads[BUFFER_COUNT];
	//! \todo factor out method here, using a template function
	for(int i=0; i < BUFFER_COUNT/THREADS_PER_BLOCK; ++i) {
		s_heads[i*THREADS_PER_BLOCK + threadIdx.x] = 0;
		s_outheads[i*THREADS_PER_BLOCK + threadIdx.x] = 0;
	}

	__shared__ uint s_synapsesPerDelay;
	__shared__ uint s_chunksPerDelay;
	__shared__ uint s_delaysPerChunk;
	if(threadIdx.x == 0) {
		//! \todo do we need to round to block size if multiple chunks per delay?
		s_synapsesPerDelay = ALIGN(sf1_maxSynapses, warpSize);
		s_chunksPerDelay = DIV_CEIL(s_synapsesPerDelay, THREADS_PER_BLOCK);
		s_delaysPerChunk = THREADS_PER_BLOCK / s_synapsesPerDelay;
	}
	__syncthreads();


	for(int preOffset=0; preOffset < partitionSize; preOffset += THREADS_PER_BLOCK) {

		__shared__ uint s_firingCount;
		//! \todo make this a re-usable chunk of memory
		__shared__ uint16_t s_firingIdx[THREADS_PER_BLOCK];
		__shared__ uint32_t s_arrivalBits[THREADS_PER_BLOCK];

		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		//! \todo load s_recentFiring here, write result to smem array
		int candidate = preOffset + threadIdx.x;
		uint32_t arrivals = s_recentFiring[candidate] & g_firingDelays[candidate];
		if(arrivals && candidate < partitionSize) {
			int nextFree = atomicAdd(&s_firingCount, 1);
			s_firingIdx[nextFree] = candidate;
			s_arrivalBits[nextFree] = arrivals;
		}
		__syncthreads();

		/* We now have the indices of the firing of THREADS_PER_BLOCK
		 * presynaptic neurons */
		for(int i=0; i<s_firingCount; ++i) {

			int presynaptic = s_firingIdx[i];

			__shared__ uint s_delayBlocks;
			__shared__ uint32_t s_arrivals[MAX_DELAY];
			if(threadIdx.x == 0) {
				s_delayBlocks = 0;
				uint32_t arrivalBits = s_arrivalBits[i];

				while(arrivalBits) {
					int arrivalDelay = __ffs(arrivalBits) - 1;
					s_arrivals[s_delayBlocks] = arrivalDelay;
					arrivalBits &= ~(0x1 << arrivalDelay);
					s_delayBlocks += 1;
				}
				s_chunkCount = s_delaysPerChunk == 0 ?
					s_delayBlocks * s_chunksPerDelay :  // >= 1 chunk(s) per delay
					DIV_CEIL(s_delayBlocks, s_delaysPerChunk);  // multiple delays per chunk
			}
			__syncthreads();

			/* The delay pitch may vary between networks, or between partitions.
			 * Even with sequential processing of presynaptic neurons, we want to
			 * maximise parallel processing of incoming spikes from different
			 * delays. We have two situations:
			 *
			 * 1) if the delay pitch is more than half the block size we process
			 *    each delay sequentially
			 * 2) if the delay pitch is less than or equal to half the block size
			 *    we process multiple delays in parallel
			 */

			for(int chunk=0; chunk < s_chunkCount; ++chunk) {

				int delayEntry = s_delaysPerChunk == 0 ?
					chunk / s_chunksPerDelay :
					threadIdx.x / s_synapsesPerDelay;
				uint32_t delay = s_arrivals[delayEntry];
				/* Offset /within/ a delay block */
				int synapseIdx = s_delaysPerChunk == 0 ?
					(chunk % s_chunksPerDelay) * THREADS_PER_BLOCK + threadIdx.x :
					(threadIdx.x % s_synapsesPerDelay);

				float weight;
				uint target;
				int bufferIdx = 0; // relative to beginning of output buffer slot
				bool doCommit = false;

				//! \todo consider using per-neuron maximum here instead
				if(synapseIdx < sf1_maxSynapses && delayEntry < s_delayBlocks
#ifdef __DEVICE_EMULATION__
						// warp size is 1, so rounding to warp size not as expected
						&& threadIdx.x < s_synapsesPerDelay * s_delaysPerChunk
#endif
				  ) {
					size_t synapseAddress =
						(presynaptic * maxDelay + delay) * f1_pitch + synapseIdx;
					weight = gf1_weights[synapseAddress];
					target = gf1_address[synapseAddress];

					if(weight != 0.0f) {
						doCommit = true;
						bufferIdx = atomicAdd(s_outheads + targetPartition(target), 1);
						//! \todo deal with STDP here
					}
				}

				/* For L1 delivery there's no race condition in the scatter
				 * step, but if care is not taken here, we get one in the
				 * gather step, as multiple spikes may converge on the same
				 * postsynaptic neuron at the same time.
				 *
				 * While we don't need to worry about race conditions, we _do_
				 * need to worry about memory bandwidth. A single firing neuron
				 * can generate spikes reaching many different targets, spread
				 * over multiple target partitions. If we naïvely deal with one
				 * firing neuron at a time and write the spike to the global
				 * memory spike queue directly, we end up with a large number
				 * of non-coalesced writes.
				 *
				 * To reduce this problem, we buffer outgoing data on a per
				 * target-partition basis. The buffers are kept as small as is
				 * reasonable (one warp) and then flushed as needed. */

				/* In the worst case, every thread writes to the same target
				 * partition, which means the buffer will easily overflow. We
				 * therefore need to interleave the filling of the output
				 * buffer with its flushing */

				//! \todo could use warp vote here, if some care is taken
				/* The number of buffers is exactly the warp size, so we
				 * can use a single warp vote to determine if we need to
				 * flush anything */
				//! \todo make this compile-time assertion
				//! \todo factor out
				__shared__ uint s_flushCount;
				__shared__ uint s_flushPartition[BUFFER_COUNT];
				do {
					/* ensure loop condition is not changed while threads are
					 * in different loop iterations */
					__syncthreads();

					/* Write one batch of data to output buffers, up to the
					 * limit of the buffer */
					if(doCommit && bufferIdx < BUFFER_SZ) {
						//! \todo do some compression here to avoid race conditions later
						s_outbuf[targetPartition(target) * BUFFER_SZ + bufferIdx] =
							make_uint2(target, __float_as_int(weight));
						doCommit = false;
						DEBUG_MSG("Buffering L1 current %f for synapse"
								"%u-?? -> %u-%u (after unknown delay)\n",
								weight, CURRENT_PARTITION,
								targetPartition(target), targetNeuron(target));
					} else {
						bufferIdx -= BUFFER_SZ; // prepare to write to buffer on subsequent loop iteration
					}

					/* Determine how many buffers are now full and need flushing */
					if(threadIdx.x == 0) {
						s_flushCount = 0;
					}
					__syncthreads();

					//! \todo factor out
					{
						ASSERT(BUFFER_COUNT <= THREADS_PER_BLOCK);
						int targetPartition = threadIdx.x;
						if(targetPartition < BUFFER_COUNT) {
							if(s_outheads[targetPartition] >= BUFFER_SZ) {
								s_outheads[targetPartition] -= BUFFER_SZ;
								uint next = atomicInc(&s_flushCount, BUFFER_COUNT);
								s_flushPartition[next] = targetPartition;
							}
						}
						__syncthreads();
					}

					/* Flush buffers */
					/*! \todo could potentially flush multiple buffers in one go here */
					for(int flush_i=0; flush_i < s_flushCount; ++flush_i) {
						int targetPartition = s_flushPartition[flush_i];
						DEBUG_THREAD_MSG(0, "Flushing buffer %d\n", targetPartition);
						flushSpikeBuffer(
                                writeBufferIdx,
								BUFFER_SZ, // flush whole buffer
								targetPartition,
								s_heads,
								s_outbuf,
								g_sq, sqPitch);
						__syncthreads();
						//! \todo could add all in parallel?
						if(threadIdx.x == 0) {
							s_heads[targetPartition] += BUFFER_SZ;
						}
					}
				} while(s_flushCount);
				__syncthreads(); // ensure every thread has left the loop, before re-entering it
			}
		}
	}

	flushAllSpikeBuffers(writeBufferIdx, headPitch, g_heads, s_heads, s_outheads, s_outbuf, g_sq, sqPitch);
	DEBUG_MSG("End deliver L1\n");
}

