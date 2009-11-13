#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

#include <stddef.h>
#include "types.h"


typedef struct Network* NETWORK;


NETWORK
cpu_set_network(fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[],
		size_t ncount,
		delay_t maxDelay);


void cpu_delete_network(NETWORK);


void cpu_add_synapses(NETWORK,
		nidx_t source,
		delay_t delay,
		nidx_t* targets,
		weight_t* weights,
		size_t length);



/*! Perform a single simulation step by delivering spikes and updating the
 * state of all neurons.
 *
 * \return
 * 		pointer to per-neuron firing vector. The return data is valid until the
 * 		next call to update.
 *
 * \param fstim
 * 		Per-neuron vector indiciating which ones should be stimulated this cycle.
 */
bool_t* cpu_step(NETWORK network, unsigned int fstim[]);



/* The step function above will do both the spike delivery and update. However,
 * it can sometimes be desirable to call these individually, e.g. for profiling
 * reasons. If so, call 'deliver_spikes', then 'update', and finally
 * 'read_firing' (optional) */

void cpu_deliver_spikes(NETWORK network);
void cpu_update(NETWORK network, unsigned int fstim[]);


/*! \return
 * 		Vector with firing (pre-neuron boolean) for the last cycle
 */
bool_t* cpu_read_firing(NETWORK network);

#endif
