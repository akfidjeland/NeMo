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
		size_t ncount)
{
	return new Network(a, b, c, d, u, v, sigma, ncount);
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



void
cpu_step(Network* network, unsigned int fstim[])
{
	network->step(fstim);
}



void
cpu_deliver_spikes(Network* network)
{
	network->deliverSpikes();
}



void
cpu_update(Network* network, unsigned int fstim[])
{
	network->update(fstim);
}



void
cpu_read_firing(Network* network,
		unsigned int** neuronIdx,
		unsigned int* nfired)
{
	const std::vector<unsigned int>& firings = network->readFiring();
	*neuronIdx = const_cast<unsigned int*>(&firings[0]);
	*nfired = firings.size();
}



long int
cpu_elapsed_ms(Network* network)
{
	return network->elapsed();
}



void
cpu_reset_timer(Network* network)
{
	network->resetTimer();
}
