/* Simple network with 1000 neurons with all-to-all connections with random
 * weights.

 * Author: Andreas K. Fidjelnad <andreas.fidjeland@imperial.ac.uk>
 * Date: April 2010
 */

#include <boost/random.hpp>

#include <nemo.hpp>
#include "sim_runner.hpp"

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;



void
addExcitatoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float a = 0.02f;
	float b = 0.2f;
	float r1 = float(param());
	float r2 = float(param());
	float c = v + 15.0f * r1 * r1;
	float d = 8.0f - 6.0f * r2 * r2;
	float u = b * v;
	float sigma = 5.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}


void
addExcitatorySynapses(nemo::Network* net, unsigned source, unsigned ncount, urng_t& rweight)
{
	std::vector<unsigned> targets(ncount, 0U);
	std::vector<unsigned> delays(ncount, 1U);
	std::vector<float> weights(ncount, 0.0f);
	std::vector<unsigned char> isPlastic(ncount, 0);

	for(unsigned target = 0; target < 1000; ++target) {
		targets.at(target) = target;
		weights.at(target) = 0.5f * float(rweight());
	}

	net->addSynapses(source, targets, delays, weights, isPlastic);
}


void
addInhibitoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float r1 = float(param());
	float a = 0.02f + 0.08f * r1;
	float r2 = float(param());
	float b = 0.25f - 0.05f * r2;
	float c = v;
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}



void
addInhibitorySynapses(nemo::Network* net, unsigned source, unsigned ncount, urng_t& rweight)
{
	std::vector<unsigned> targets(ncount, 0);
	std::vector<unsigned> delays(ncount, 1U);
	std::vector<float> weights(ncount, 0.0f);
	std::vector<unsigned char> isPlastic(ncount, 0);

	for(unsigned target = 0; target < ncount; ++target) {
		targets.at(target) = target;
		weights.at(target) = float(-rweight());
	}

	net->addSynapses(source, targets, delays, weights, isPlastic);
}




nemo::Network*
construct(unsigned ncount)
{
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));

	nemo::Network* net = new nemo::Network();

	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		if(nidx < (ncount * 4) / 5) { // excitatory
			addExcitatoryNeuron(net, nidx, randomParameter);
			addExcitatorySynapses(net, nidx, ncount, randomParameter);
		} else { // inhibitory
			addInhibitoryNeuron(net, nidx, randomParameter);
			addInhibitorySynapses(net, nidx, ncount, randomParameter);
		}
	}
	return net;
}



int
main(int argc, char* argv[])
{
	unsigned ncount = 1000;

	nemo::Network* net = construct(ncount);
	nemo::Configuration conf;
	nemo::Simulation* sim = nemo::Simulation::create(*net, conf);
	if(sim == NULL) {
		std::cerr << "failed to create simulation" << std::endl;
		return -1;
	}
	//simulate(sim, ncount, ncount);
	simulateToFile(sim, 1000, "firing.dat");
	delete net;
	return 0;
}
