#ifndef KERNEL_HPP
#define KERNEL_HPP

/*! \file kernel.hpp Prototypes for kernel calls */

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "outgoing.cu_h"
#include "globalQueue.cu_h"
#include "localQueue.cu_h"
#include "types.h"
#include "bitvector.cu_h"

/*! Update synapse weight using the accumulated eligibility trace
 *
 * \param minExcitatoryWeight lowest \i positive weight for excitatory synapses
 * \param maxExcitatoryWeight highest \i positive weight for excitatory synapses
 * \param minInhibitoryWeight lowest absolute \i negative weight for inhibitory synapses
 * \param maxInhibitoryWeight highest absolute \i negative weight for inhibitory synapses
 *
 * Synapses 'stick' at zero. If the minimum weight is non-zero, this will never
 * occur.
 *
 * \pre 0 <= minExcitatoryWeight < maxExcitatoryWeight
 * \pre 0 >= minInhibitoryWeight < maxInhibitoryWeight
 */
void
applyStdp(
		cudaStream_t stream,
		unsigned partitionCount,
		unsigned fractionalBits,
		synapse_t* d_fcm,
		float minExcitatoryWeight,
		float maxExcitatoryWeight,
		float minInhibitoryWeight,
		float maxInhibitoryWeight,
		float reward);


cudaError_t
gather( cudaStream_t stream,
		unsigned partitionCount,
		unsigned cycle,
		fix_t* d_current,
		synapse_t* d_fcm,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill);



cudaError_t
fire(	cudaStream_t stream,
		unsigned partitionCount,
		unsigned cycle,
		bool thalamicInputEnabled,
		float* df_neuronParameters,
		float* df_neuronState,
		unsigned* du_neuronState,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		fix_t* d_istim,
		fix_t* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired);


cudaError_t
scatter(cudaStream_t stream,
		unsigned partitionCount,
		unsigned cycle,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		outgoing_addr_t* d_outgoingAddr,
		outgoing_t* d_outgoing,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill,
		lq_entry_t* d_lqData,
		unsigned* d_lqFill,
		uint64_t* d_delays);


cudaError_t
updateStdp(
		cudaStream_t stream,
		unsigned partitionCount,
		unsigned cycle,
		uint64_t* d_recentFiring,
		uint32_t* d_dfired,
		unsigned* d_nFired,
		nidx_dt* d_fired);


cudaError
configureStdp(
		unsigned preFireWindow,
		unsigned postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		weight_dt* stdpFn);

cudaError configurePartitionSize(const unsigned* d_partitionSize, size_t len);

cudaError fx_setFormat(unsigned fractionalBits);
cudaError bv_setPitch(size_t pitch);

void initLog();
void flushLog();
void endLog();

#endif
