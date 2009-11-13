extern "C" {
#include "cpu_kernel.h"
}

#include "Network.hpp"


struct Network*
cpu_set_network(fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[], //set to 0 if not thalamic input required
		size_t ncount,
		delay_t maxDelay)
{
	return new Network(a, b, c, d, u, v, sigma, ncount, maxDelay);
}



void
cpu_add_synapses(Network* net,
		nidx_t source,
		delay_t delay,
		nidx_t* targets,
		weight_t* weights,
		size_t length)
{
	net->setCMRow(source, delay, targets, weights, length);
}



void
cpu_delete_network(Network* net)
{
	delete net; 
}



bool_t*
cpu_step(Network* network, unsigned int fstim[])
{
	network->step(fstim);
}



bool_t*
cpu_update(Network* network, unsigned int fstim[])
{
	network->update(fstim);
}



void
cpu_deliver_spikes(Network* network)
{
	network->deliverSpikes();
}
