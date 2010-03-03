#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

#include <stddef.h>
#include "types.h"


// opaque pointer
typedef void* NETWORK;


/* Create a new empty network. Use cpu_add_neuron and cpu_add_synapse to
 * construct the network */
NETWORK
cpu_new_network();


/* Create a new network with all the neurons initialized. Connectivity can be
 * added using cpu_add_synapses. */
NETWORK
cpu_set_network(fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[],
		size_t ncount);


cpu_status_t
cpu_enable_stdp(NETWORK,
		double* pre_fn,
		size_t pre_len,
		double* post_fn,
		size_t post_len,
		double minWeight,
		double maxWeight);


cpu_status_t
cpu_start_simulation(NETWORK);


cpu_status_t
cpu_add_neuron(NETWORK,
		nidx_t idx,
		fp_t a, fp_t b, fp_t c, fp_t d,
		fp_t u, fp_t v, fp_t sigma);


cpu_status_t
cpu_add_synapses(NETWORK,
		nidx_t source,
		delay_t delay,
		nidx_t targets[],
		weight_t weights[],
		unsigned int is_plastic[],
		size_t length);



/*! Perform a single simulation step by delivering spikes and updating the
 * state of all neurons. No firings are returned. Instead these are read using
 * cpu_read_firing.
 *
 * \param fstim
 * 		Per-neuron vector indiciating which ones should be stimulated (forced
 * 		to fire) this cycle.
 */
cpu_status_t cpu_step(NETWORK network, unsigned int fstim[]);



/*!
 * The step function above will do both the spike delivery and update. However,
 * it can sometimes be desirable to call these individually, e.g. if profiling.
 * If so, call 'deliver_spikes', then 'update'.
 */
cpu_status_t cpu_deliver_spikes(NETWORK network);
cpu_status_t cpu_update(NETWORK network, unsigned int fstim[]);



/*!
 * Return list of fired neurons for the most recent simulation cycle.
 *
 * \param fired
 * 		indices of neurons which fired. This array should not be modified, and
 * 		is safe to access until the next call to cpu_step.
 * \param nfired
 * 		number of neurons which fired
 */
cpu_status_t
cpu_read_firing(NETWORK network,
		unsigned int** fired,
		unsigned int* nfired);



cpu_status_t
cpu_apply_stdp(NETWORK network, double reward);


/*! \return number of milliseconds elapsed between beginning of first kernel
 * invocation and the end of the last */
long int cpu_elapsed_ms(NETWORK);


cpu_status_t cpu_reset_timer(NETWORK);

void cpu_delete_network(NETWORK);


/* If a runtime error occurred (indicated via cpu_status_t return of some other
 * function), this function returns a description of the last error */
const char* cpu_last_error(NETWORK);

#endif
