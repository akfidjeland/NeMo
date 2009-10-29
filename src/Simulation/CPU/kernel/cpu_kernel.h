#ifndef CPU_KERNEL_H
#define CPU_KERNEL_H

typedef struct Network* NETWORK;
typedef unsigned int bool_t;

NETWORK
set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[],
		unsigned int len);

void delete_network(NETWORK);


/* Update the state of all neurons, returning pointer to per-neuron firing
 * vector. The return data is valid until the next call to update. */
bool_t* update(NETWORK network, unsigned int[], double current[]);


#endif
