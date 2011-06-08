#ifndef NEMO_CUDA_CURRENT_CU
#define NEMO_CUDA_CURRENT_CU

/*! \file current.cu Functions related to neuron input current */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "fixedpoint.cu"


/*! \brief Add input current for a particular neuron
 *
 * The input current is stored in shared memory in a fixed-point format. This
 * necessitates overflow detection, so that we can use saturating arithmetic.
 *
 * \param[in] neuron
 *		0-based index of the target neuron
 * \param[in] current
 *		current in mA in fixed-point format
 * \param s_current
 *		shared memory vector containing current for all neurons in partition
 * \param[out] s_overflow
 *		bit vector indicating overflow status for all neurons in partition
 * \param[out] s_negative
 *		bit vector indicating the overflow sign for all neurons in partition
 *
 * \pre neuron < partition size
 * \pre all shared memory buffers have at least as many entries as partition size
 *
 * \todo add cross-reference to fixed-point format documentation
 */
__device__
void
addCurrent(nidx_t neuron,
		fix_t current,
		fix_t* s_current,
		uint32_t* s_overflow,
		uint32_t* s_negative)
{
	ASSERT(neuron < MAX_PARTITION_SIZE);
	bool overflow = fx_atomicAdd(s_current + neuron, current);
	bv_atomicSetPredicated(overflow, neuron, s_overflow);
	bv_atomicSetPredicated(overflow && fx_isNegative(current), neuron, s_negative);
#ifndef FIXPOINT_SATURATION
	ASSERT(!overflow);
#endif
}



/*! \brief Add externally provided current stimulus
 *
 * The user can provide per-neuron current stimulus
 * (\ref nemo::cuda::Simulation::addCurrentStimulus).
 *
 * \param[in] psize
 *		number of neurons in current partition
 * \param[in] pitch
 *		pitch of g_current, i.e. distance in words between each partitions data
 * \param[in] g_current
 *		global memory vector containing current for all neurons in partition.
 *		If set to NULL, no input current will be delivered.
 * \param s_current
 *		shared memory vector containing current for all neurons in partition
 * \param s_overflow
 *		bit vector indicating overflow status for all neurons in partition.
 *		Entries here may already be set and are simply OR-ed with any new entries.
 * \param s_negative
 *		bit vector indicating the overflow sign for all neurons in partition
 *		Entries here may already be set and are simply OR-ed with any new entries.
 *
 * \pre neuron < size of current partition
 * \pre all shared memory buffers have at least as many entries as the size of
 * 		the current partition
 *
 * \see nemo::cuda::Simulation::addCurrentStimulus
 */
__device__
void
addCurrentStimulus(unsigned psize,
		size_t pitch,
		const fix_t* g_current,
		fix_t* s_current,
		uint32_t* s_overflow,
		uint32_t* s_negative)
{
	if(g_current != NULL) {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			unsigned pstart = CURRENT_PARTITION * pitch;
			fix_t stimulus = g_current[pstart + neuron];
			addCurrent(neuron, stimulus, s_current, s_overflow, s_negative);
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
			DEBUG_MSG_SYNAPSE("c%u %u-%u: +%f (external)\n",
					s_cycle, CURRENT_PARTITION, neuron,
					fx_tofloat(g_current[pstart + neuron]));
#endif
		}
		__syncthreads();
	}
}



/*! Copy per-neuron accumulated current between two memory areas
 *
 * \param[in] current_in Per-neuron accumulated current (shared or global memory)
 * \param[out] current_out Per-neuron accumulated current (shared or global memory)
 *
 * Global memory arguments must be offset to the appropriate partition.
 */
__device__
void
copyCurrent(unsigned nNeurons, fix_t* current_in, fix_t* current_out)
{
	for(unsigned bNeuron=0; bNeuron < nNeurons; bNeuron += THREADS_PER_BLOCK) {
		unsigned neuron = bNeuron + threadIdx.x;
		current_out[neuron] = current_in[neuron];
	}
}

#endif
