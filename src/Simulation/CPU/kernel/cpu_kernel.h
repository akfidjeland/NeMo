#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

#include <stddef.h>
#include "types.h"

//#define DEBUG_TRACE

typedef struct Network* NETWORK;


//! \todo use weight_t here as well? Keep a consisten floating point type
NETWORK
set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[],
		//! \todo use size_t for consistency here
		unsigned int ncount,
		delay_t maxDelay);

void delete_network(NETWORK);


void add_synapses(NETWORK,
		nidx_t source,
		delay_t delay,
		nidx_t* targets,
		weight_t* weights,
		size_t length);


/*! Update the state of all neurons, returning pointer to per-neuron firing
 * vector. The return data is valid until the next call to update.
 *
 * \param fstim
 * 		Per-neuron vector indiciating which ones should be stimulated this cycle.
 */
bool_t* update(NETWORK network, unsigned int fstim[]);


#endif
