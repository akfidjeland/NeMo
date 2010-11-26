/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "localQueue.cu_h"

/* Local queue pitch in terms of words for each delay/partition pair */
__constant__ size_t c_lqPitch;


__host__
cudaError
setLocalQueuePitch(size_t pitch)
{
	return cudaMemcpyToSymbol(c_lqPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}


/*! \return offset into local queue fill data (in gmem) for the current
 * partition and the given slot. */
__device__
unsigned
lq_globalFillOffset(unsigned slot)
{
	return CURRENT_PARTITION * MAX_DELAY + slot;
}


/*! \return queue fill for the current partition slot due for delivery now, and
 * reset the relevant slot. */
__device__
unsigned
lq_getAndClearCurrentFill(unsigned cycle, unsigned* g_fill)
{
	return atomicExch(g_fill + lq_globalFillOffset(cycle % MAX_DELAY), 0); 
}



/*! Load current queue fill from gmem to smem. 
 * 
 * \pre s_fill contains at least MAX_DELAY elements 
 */ 
__device__
void
lq_loadQueueFill(unsigned* g_fill, unsigned* s_fill)
{
	unsigned slot = threadIdx.x;
	if(slot < MAX_DELAY) {
		s_fill[slot] = g_fill[lq_globalFillOffset(slot)];
	}
}


/*! Store updated queue fill back to gmem */
__device__
void
lq_storeQueueFill(unsigned* s_fill, unsigned* g_fill)
{
	unsigned slot = threadIdx.x;
	if(slot < MAX_DELAY) {
		g_fill[lq_globalFillOffset(slot)] = s_fill[slot];
	}
}



/*! \return the buffer number to use for the given delay, given current cycle */
__device__
unsigned
lq_delaySlot(unsigned cycle, unsigned delay0)
{
	return (cycle + delay0) % MAX_DELAY;
}


/*! \return the full address to the start of a queue entry (for a
 * partition/delay pair) given a precomputed slot number */
__device__
unsigned
lq_offsetOfSlot(unsigned slot)
{
	ASSERT(slot < MAX_DELAY);
	return (CURRENT_PARTITION * MAX_DELAY + slot) * c_lqPitch;
}


/*! \return the full address to the start of a queue entry for the current
 * partition and the given delay (relative to the current cycle). */
__device__
unsigned
lq_offset(unsigned cycle, unsigned delay0)
{
	return lq_offsetOfSlot(lq_delaySlot(cycle, delay0));
}



/*! \return offset to next free queue slot in gmem queue for current partition
 * 		and the given \a delay0 offset from the given \a cycle.
 *
 * \param cycle
 * \param delay0
 * \param s_fill shared memory buffer which should have been previously filled
 *        using lq_loadQueueFill
 */
__device__
unsigned
lq_nextFree(unsigned cycle, delay_t delay0, unsigned* s_fill)
{
	/* The buffer should be sized such that we never overflow into the next
	 * queue slot (or out of the queue altogether). However, even if this is
	 * not the case the wrap-around in the atomic increment ensures that we
	 * just overwrite our own firing data rather than someone elses */
	unsigned delaySlot = lq_delaySlot(cycle, delay0);
	ASSERT(delaySlot < MAX_DELAY);
	ASSERT(lq_offsetOfSlot(delaySlot) < PARTITION_COUNT * MAX_DELAY * c_lqPitch);
	unsigned next = atomicInc(s_fill + delaySlot, c_lqPitch-1);
	ASSERT(next < c_lqPitch);
	return lq_offsetOfSlot(delaySlot) + next;
}



/*! Enqueue a single neuron/delay pair in the local queue
 *
 * This operation will almost certainly be non-coalesced.
 */
__device__
void
lq_enque(nidx_dt neuron,
		unsigned cycle,
		delay_t delay0,
		unsigned* s_fill,
		lq_entry_t* g_queue)
{
	g_queue[lq_nextFree(cycle, delay0, s_fill)] = make_short2(neuron, delay0);
}
