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

#include <stdint.h>
#include <stddef.h>

#include "STDP.hpp"
#include "outgoing.cu_h"
#include "incoming.cu_h"
#include "nemo_cuda_types.h"

void
applyStdp(
		unsigned long long* d_cc,
		size_t ccPitch,
		uint partitionCount,
		uint fractionalBits,
		synapse_t* d_fcm,
		const nemo::STDP<float>& stdpFn,
		float reward);


void
stepSimulation(
		uint partitionCount,
		bool usingStdp,
		uint cycle,
		uint64_t* d_recentFiring,
		float* d_neuronState,
		unsigned* d_rngState,
		float* d_rngSigma,
		uint32_t* d_fstim,
		uint32_t* d_fout,
		synapse_t* d_fcm,
		uint* d_outgoingCount,
		outgoing_t* d_outgoing,
		uint* d_incomingHeads,
		incoming_t* d_incoming,
		unsigned long long* d_cc,
		size_t ccPitch);

void
configureStdp(
		uint preFireWindow,
		uint postFireWindow,
		uint64_t potentiationBits,
		uint64_t depressionBits,
		weight_dt* stdpFn);

#endif
