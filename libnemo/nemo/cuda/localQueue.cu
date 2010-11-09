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
 * partition and the given delay. */
__device__
unsigned
lq_globalFillOffset(unsigned delay0)
{
	return CURRENT_PARTITION * MAX_DELAY + delay0;
}


/*! Load current queue fill from gmem to smem. 
 * 
 * \pre s_fill contains at least MAX_DELAY elements 
 */ 
__device__
void
lq_loadQueueFill(unsigned* g_fill, unsigned* s_fill)
{
	unsigned delay0 = threadIdx.x;
	if(delay0 < MAX_DELAY) {
		s_fill[delay0] = g_fill[lq_globalFillOffset(delay0)];
	}
}


/*! Store updated queue fill back to gmem */
__device__
void
lq_storeQueueFill(unsigned* s_fill, unsigned* g_fill)
{
	unsigned delay0 = threadIdx.x;
	if(delay0 < MAX_DELAY) {
		g_fill[lq_globalFillOffset(delay0)] = s_fill[delay0];
	}
}



/*! \return the buffer number to use for the given delay, given current cycle */
__device__
unsigned
lq_delaySlot(unsigned cycle, unsigned delay1)
{
	return (cycle + delay1) % MAX_DELAY;
}



/* Return offset to next free queue slot in gmem queue for current partition
 * and the given delay
 *
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
	unsigned delaySlot = lq_delaySlot(cycle, delay0+1);
	return (CURRENT_PARTITION * MAX_DELAY + delaySlot) * c_lqPitch 
		+ atomicInc(s_fill + delaySlot, c_lqPitch-1);
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
