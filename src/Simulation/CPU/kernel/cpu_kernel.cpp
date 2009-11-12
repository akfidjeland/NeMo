extern "C" {
#include "cpu_kernel.h"
}

#include "Network.hpp"


struct Network*
cpu_set_network(double a[],
		double b[],
		double c[],
		double d[],
		double u[],
		double v[],
		double sigma[], //set to 0 if not thalamic input required
		unsigned int ncount,
		delay_t maxDelay)
{
	return new Network(a, b, c, d, u, v, sigma, ncount, maxDelay);
}



void
cpu_add_synapses(NETWORK net,
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
