#ifndef APPLY_STDP_CU
#define APPLY_STDP_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda.h>
#include "cycleCounting.cu"
#include "connectivityMatrix.cu"
#include "util.h"
#include "fixedpoint.cu"


/*! Apply STDP 
 * 
 * The STDP statistics are stored in reverse CM order with potentiation and
 * depression already combined. This data needs to be re-ordered into the
 * forward order when updating the weight.
 *
 * The new weight is limited by a maximum weight, and is not allowed to fall
 * below 0.
 *
 * prefix r: reverse matrix
 * prefix f: forward matrix
 */
__global__
void
applySTDP_(
#ifdef KERNEL_TIMING
	unsigned long long* g_cc,
	size_t ccPitch,
#endif
	synapse_t* g_fcm,
	weight_dt reward,
	weight_dt maxWeight, // for excitatory synapses
	weight_dt minWeight) // for inhibitory synapses
	/*! \note reverse connectivity addresses are found in constant memory,
	 * while forward connectivity addresses are found in texture memory */
{
	SET_COUNTER(s_ccApplySTDP, 0);

	__shared__ uint s_chunkCount;
	__shared__ uint s_partitionSize;

	weight_dt* gr_stdp = (weight_dt*) cr_stdp[CURRENT_PARTITION];
	uint r_pitch = cr_pitch[CURRENT_PARTITION];

#if __DEVICE_EMULATION__
	uint32_t* gr_address = (uint32_t*) cr_address[CURRENT_PARTITION];
#endif

	uint32_t* gr_faddress = (uint32_t*) cr_faddress[CURRENT_PARTITION];

	if(threadIdx.x == 0) {
		s_partitionSize = c_partitionSize[CURRENT_PARTITION];
		s_chunkCount = DIV_CEIL(r_pitch, THREADS_PER_BLOCK);
	}
	__syncthreads();

	for(uint target=0; target < s_partitionSize; ++target) {
		for(uint chunk=0; chunk < s_chunkCount; ++chunk) {

			uint r_sidx = chunk * THREADS_PER_BLOCK + threadIdx.x;

			if(r_sidx < r_pitch) {

				size_t gr_offset = target * r_pitch + r_sidx;
				size_t gf_offset = gr_faddress[gr_offset];
#if __DEVICE_EMULATION__
				//! \todo remove loading of rsynapse
				uint rsynapse = gr_address[gr_offset];
#endif

				if(gf_offset != 0) {

					/*! \todo try using atomicExch here instead. For m=20
					 * atomicExch is slightly faster, but this will probably
					 * work less well for e.g. m=1000 */
					//weight_dt w_diff = gr_stdp[gr_offset] * reward;
					//float w_diff = reward * __int_as_float(atomicExch(gr_stdp + gr_offset, __float_as_int(0.0f)));
					weight_dt w_diff = fx_mul(gr_stdp[gr_offset], reward);

					if(w_diff != 0) {

						gr_stdp[gr_offset] = 0;

						weight_dt* gf_weight = (weight_dt*) g_fcm + c_fcmPlaneSize * FCM_WEIGHT;

						weight_dt w_old = gf_weight[gf_offset];
						weight_dt w_new = 0;
						if(w_old > 0) {
							w_new = min(maxWeight, max(w_old + w_diff, 0));
						} else if(w_old < 0) {
							w_new = min(0, max(w_old + w_diff, minWeight));
						}

						if(w_old != w_new) {
							gf_weight[gf_offset] = w_new;
							DEBUG_MSG("stdp (%u-%u -> %u-%u) %f %+f = %f\n",
									sourcePartition(rsynapse), sourceNeuron(rsynapse),
									CURRENT_PARTITION, target,
									fx_tofloat(w_old), fx_tofloat(w_diff), fx_tofloat(w_new));
						}
					}
				}
			}
		}
		//! \todo remove sync?
		__syncthreads();
	}

	SET_COUNTER(s_ccApplySTDP, 1);
	WRITE_COUNTERS(s_ccApplySTDP, g_cc, ccPitch, 2);
}

#endif
