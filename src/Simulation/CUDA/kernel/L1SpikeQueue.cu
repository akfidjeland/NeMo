#include "kernel.cu_h"
#include "util.h"
#include <stdint.h>
#ifdef __DEVICE_EMULATION__
#   include <assert.h>
#   include <stdio.h>
#endif

#undef STDP_FN
#ifdef STDP
#define STDP_FN(f) f ## _STDP
#else
#define STDP_FN(f) f ## _static
#endif




/*! Update current buffer with incoming spikes */
__device__
void
STDP_FN(updateCurrent)(
		uint readBufferIdx, // double buffer index
		uint sourcePartition,
		uint32_t spikeIdx,   // index of spike for current thread
		uint32_t spikeCount, // number of spikes to delivered for current buffer 
		uint2* g_sb,
		size_t sbPitch,
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
	size_t base = g_sbBase(sbPitch, sourcePartition, targetPartition, readBufferIdx);
	//! \todo load by 32-bit values
	uint2 spike = g_sb[base + spikeIdx];

	/* We don't need to clear the data from the spike buffer,
	 * as long as the head is cleared. \see loadAndClearBufferHeads */
	float weight = spikeWeight(spike);
	uint target = spikeTargetNeuron(spike);

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

	//! \todo share commit loop with L1 delivery?
	for(uint commit=0; commit <= s_maxCommit; ++commit) {
		if(spiked && commitNo == commit) {
			//! \todo use target from earlier
			s_current[targetNeuron(spike.x)] += weight;
			ASSERT(targetNeuron(spike.x) < MAX_PARTITION_SIZE);
			DEBUG_MSG("Receiving L1 current %f from %d-?? to %d-%d\n",
					weight, sourcePartition, targetPartition, targetNeuron(spike.x));
		}
		__syncthreads();
	}
	__syncthreads();
}



/*! Load all incoming spikes from L1 connectivity into current accumulator */
__device__
void
STDP_FN(gatherL1Spikes_JIT_)(
		uint readBufferIdx,
		uint2* g_sb,
		size_t sbPitch,
		uint* g_heads,
        size_t headPitch,
		float* s_current,
		uint* s_heads /* s_P32 */ )
{
	loadAndClearBufferHeads(g_heads, s_heads, headPitch, readBufferIdx);
	for(uint src=0; src<PARTITION_COUNT; ++src) {
		uint parallelLoads = DIV_CEIL(s_heads[src], THREADS_PER_BLOCK);
		for(uint load=0; load<parallelLoads; ++load) {
			uint spikeIdx = load * THREADS_PER_BLOCK + threadIdx.x;
			STDP_FN(updateCurrent)(readBufferIdx, src, spikeIdx, s_heads[src],
					g_sb, sbPitch, s_current);
		}
	}
	__syncthreads();
}


/* TODO: the loop structure here is nearly the same as deliverL0Spikes. Factor
 * out or use a code generator to avoid repetition */
__device__
void
STDP_FN(deliverL1Spikes_JIT)(
	uint maxDelay,
	uint writeBufferIdx,
	uint partitionSize,
	uint sf1_maxSynapses,
	uint* gf1_cm, uint f1_pitch, uint f1_size,
	uint32_t* s_recentFiring,
	uint32_t* g_firingDelays,
	// L1 spike queue
    //! \todo allow more than 32 partitions (by splitting L1CM)
	uint2* s_outbuf,        // 16 words of buffer per target partition
	uint2* g_sb,
	size_t sbPitch,
	uint* g_heads,
	size_t headPitch,
	uint16_t* s_firingIdx,
	uint32_t* s_arrivalBits,
	uint32_t* s_arrivals,
	/* In global memory there is one buffer per source-target partition pair */
	uint32_t* s_gheads /* s_P32 */)
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
	 * writes we stage spike data in shared memory before writing it to global
	 * memory.
	 */

	/* For the output buffers, multiple buffers may be dedicated to the same
	 * global buffer. This is to avoid a sequential bottleneck when we have
	 * only a small number of partitions */

	/*! \todo use one of the general-purpose chunks of shared memory */
	__shared__ uint s_sheads[MAX_BUFFER_COUNT];

	/* For both s_gheads and s_sheads, the data are small enough that uint2
	 * would work. However, we need to perform atomic operations, and therefore
	 * need 32-bit data */

	/* Since spikes are delivered just-in-time, the buffer is only written to
	 * during one cycle. The buffer is thus cleared before each delivery, and
	 * there's no need to load global buffer heads in the source partition. */
	s_clear<MAX_PARTITION_COUNT>(s_gheads);
	s_clear<MAX_BUFFER_COUNT>(s_sheads);

	/* Depending on the number of partitions, not all output buffers may be in
	 * use due to rounding effects */
	__shared__ uint s_bufferCount;
	__shared__ uint s_buffersPerPartition;

	__shared__ uint s_synapsesPerDelay;
	__shared__ uint s_chunksPerDelay;
	__shared__ uint s_delaysPerChunk;
	if(threadIdx.x == 0) {
		//! \todo do we need to round to block size if multiple chunks per delay?
#ifdef __DEVICE_EMULATION__
		s_synapsesPerDelay = ALIGN(sf1_maxSynapses, 32);
#else
		s_synapsesPerDelay = ALIGN(sf1_maxSynapses, warpSize);
#endif
		s_chunksPerDelay = DIV_CEIL(s_synapsesPerDelay, THREADS_PER_BLOCK);
		s_delaysPerChunk = THREADS_PER_BLOCK / s_synapsesPerDelay;
		s_buffersPerPartition = s_sbBuffersPerPartition();
		s_bufferCount = s_sbCount();
	}
	__syncthreads();


	for(int preOffset=0; preOffset < partitionSize; preOffset += THREADS_PER_BLOCK) {

		__shared__ uint s_firingCount;

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
			for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

				uint delayEntry = s_delaysPerChunk == 0 ?
					chunk / s_chunksPerDelay :
					threadIdx.x / s_synapsesPerDelay;
				uint32_t delay = s_arrivals[delayEntry];
				/* Offset /within/ a delay block */
				uint synapseIdx = s_delaysPerChunk == 0 ?
					(chunk % s_chunksPerDelay) * THREADS_PER_BLOCK + threadIdx.x :
					(threadIdx.x % s_synapsesPerDelay);

				float weight;
				uint target;
				uint bufferOffset = 0; // relative to beginning of output buffer slot
				uint bufferIdx = 0;
				bool doCommit = false;

				//! \todo consider using per-neuron maximum here instead
				if(synapseIdx < sf1_maxSynapses && delayEntry < s_delayBlocks) {
					size_t synapseAddress =
						(presynaptic * maxDelay + delay) * f1_pitch + synapseIdx;
					weight = gf1_weights[synapseAddress];

					if(weight != 0.0f) {
						doCommit = true;
						target = gf1_address[synapseAddress];
						bufferIdx =
							s_sbBufferIdx(targetPartition(target),
								s_buffersPerPartition);
						bufferOffset = atomicAdd(s_sheads + bufferIdx, 1);
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

				//! \todo factor out
				__shared__ uint s_flushCount;
				//! \todo use shared memory for this
				__shared__ uint s_flushPartition[MAX_BUFFER_COUNT];
				do {
					/* ensure loop condition is not changed while threads are
					 * in different loop iterations */
					__syncthreads();

					/* Write one batch of data to output buffers, up to the
					 * limit of the buffer */
					if(doCommit && bufferOffset < BUFFER_SZ) {
						//! \todo do some compression here to avoid race conditions later
						s_outbuf[bufferIdx * BUFFER_SZ + bufferOffset] =
							STDP_FN(packSpike)(
									presynaptic,
									delay,
									synapseIdx,
									targetNeuron(target),
									weight);
						doCommit = false;
						DEBUG_MSG("Buffering L1 current %f for synapse "
								"%u-%u -> %u-%u (after unknown delay, buffer %u[%u])\n",
								weight, CURRENT_PARTITION, presynaptic,
								targetPartition(target), targetNeuron(target),
								bufferIdx, bufferOffset);
					} else {
						bufferOffset -= BUFFER_SZ; // prepare to write to buffer on subsequent loop iteration
					}

					/* Determine how many buffers are now full and need flushing */
					if(threadIdx.x == 0) {
						s_flushCount = 0;
					}
					__syncthreads();

					//! \todo factor out
					{
						ASSERT(MAX_BUFFER_COUNT <= THREADS_PER_BLOCK);
						uint bufferIdx = threadIdx.x;
						if(bufferIdx < s_bufferCount) {
							if(s_sheads[bufferIdx] >= BUFFER_SZ) {
								s_sheads[bufferIdx] -= BUFFER_SZ;
								uint next = atomicInc(&s_flushCount, MAX_BUFFER_COUNT);
								s_flushPartition[next] = bufferIdx;
							}
						}
						__syncthreads();
					}

					/* Flush buffers */
					for(uint flush_i=0; flush_i < s_flushCount; flush_i += BUFFERS_PER_BLOCK) {
						uint i = flush_i + threadIdx.x / THREADS_PER_BUFFER;
						bool validBuffer = i < s_flushCount;
						s_sbFlush(
								validBuffer,
								s_flushPartition[i],
								writeBufferIdx,
								s_buffersPerPartition,
								BUFFER_SZ, // flush whole buffer
								s_gheads,
								s_outbuf,
								g_sb, sbPitch);
					}
				} while(s_flushCount);
				/* ensure every thread has left the loop, before re-entering it */
				__syncthreads();
			}
		}
	}

	s_sbFlushAll(
			s_buffersPerPartition,
			writeBufferIdx,
			headPitch, g_heads, s_gheads, s_sheads, s_outbuf, g_sb, sbPitch);
}

