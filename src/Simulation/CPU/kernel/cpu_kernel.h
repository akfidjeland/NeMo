#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

#include <stddef.h>
#include "types.h"

//#define DEBUG_TRACE

typedef struct Network* NETWORK;


//! \todo use weight_t here as well? Keep a consisten floating point type
NETWORK
cpu_set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[],
		//! \todo use size_t for consistency here
		unsigned int ncount,
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
 * reasons. If so, call 'deliver_spikes', then 'update' */

void cpu_deliver_spikes(NETWORK network);
bool_t* update(NETWORK network, unsigned int fstim[]);



#endif
