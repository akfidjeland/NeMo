extern "C" {
#include "cpu_kernel.h"
}

#include "Network.hpp"



NETWORK
cpu_new_network()
{
	return new nemo::cpu::Network();
}



NETWORK
cpu_set_network(fp_t a[],
		fp_t b[],
		fp_t c[],
		fp_t d[],
		fp_t u[],
		fp_t v[],
		fp_t sigma[], //set to 0 if not thalamic input required
		size_t ncount)
{
	return new nemo::cpu::Network(a, b, c, d, u, v, sigma, ncount);
}



void
cpu_add_neuron(NETWORK net,
		nidx_t idx,
		fp_t a, fp_t b, fp_t c, fp_t d,
		fp_t u, fp_t v, fp_t sigma)
{
	static_cast<nemo::cpu::Network*>(net)->addNeuron(idx, a, b, c, d, u, v, sigma);
}



void
cpu_add_synapses(NETWORK net,
		nidx_t source,
		delay_t delay,
		nidx_t* targets,
		weight_t* weights,
		size_t length)
{
	static_cast<nemo::cpu::Network*>(net)->addSynapses(source, delay, targets, weights, length);
}



void
cpu_start_simulation(NETWORK network)
{
	static_cast<nemo::cpu::Network*>(network)->startSimulation();
}



void
cpu_step(NETWORK network, unsigned int fstim[])
{
	static_cast<nemo::cpu::Network*>(network)->step(fstim);
}



void
cpu_deliver_spikes(NETWORK network)
{
	static_cast<nemo::cpu::Network*>(network)->deliverSpikes();
}



void
cpu_update(NETWORK network, unsigned int fstim[])
{
	static_cast<nemo::cpu::Network*>(network)->update(fstim);
}



void
cpu_read_firing(NETWORK network,
		unsigned int** neuronIdx,
		unsigned int* nfired)
{
	const std::vector<unsigned int>& firings =
		static_cast<nemo::cpu::Network*>(network)->readFiring();
	*neuronIdx = const_cast<unsigned int*>(&firings[0]);
	*nfired = firings.size();
}



long int
cpu_elapsed_ms(NETWORK network)
{
	return static_cast<nemo::cpu::Network*>(network)->elapsed();
}



void
cpu_reset_timer(NETWORK network)
{
	(static_cast<nemo::cpu::Network*>(network))->resetTimer();
}



void
cpu_delete_network(NETWORK net)
{
	delete static_cast<nemo::cpu::Network*>(net);
}



