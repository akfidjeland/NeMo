#ifndef KERNEL_HPP
#define KERNEL_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "outgoing.cu_h"
#include "incoming.cu_h"
#include "localQueue.cu_h"
#include "types.h"

void
applyStdp(
		unsigned long long* d_cc,
		size_t ccPitch,
		unsigned partitionCount,
		unsigned fractionalBits,
		synapse_t* d_fcm,
		float maxWeight,
		float minWeight,
		float reward);


void
stepSimulation(
		unsigned partitionCount,
		bool stdpEnabled,
		bool thalamicInputEnabled,
		unsigned cycle,
		uint64_t* d_recentFiring,
		float* df_neuronParameters,
		float* df_neuronState,
		unsigned* du_neuronState,
		uint32_t* d_fstim,
		fix_t* d_istim,
		uint32_t* d_fout,
		synapse_t* d_fcm,
		unsigned* d_outgoingCount,
		outgoing_t* d_outgoing,
		unsigned* d_incomingHeads,
		incoming_t* d_incoming,
		lq_entry_t* d_lqData,
		unsigned* d_lqFill,
		uint64_t* d_delays,
		unsigned long long* d_cc,
		size_t ccPitch);

cudaError
configureStdp(
		unsigned preFireWindow,
		unsigned postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		weight_dt* stdpFn);

cudaError configurePartitionSize(const unsigned* d_partitionSize, size_t len);

cudaError fx_setFormat(unsigned fractionalBits);

void initLog();
void flushLog();
void endLog();

#endif
