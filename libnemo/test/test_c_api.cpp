#define BOOST_TEST_MODULE nemo test_c_api

#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include <nemo.h>

#include "utils.hpp"

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

void
addExcitatoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned nidx, urng_t& param)
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
	nemo_add_neuron(c_net, nidx, a, b, c, d, u, v, sigma);
}


void
addExcitatorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	std::vector<unsigned> targets(scount, 0U);
	std::vector<unsigned> delays(scount, 1U);
	std::vector<float> weights(scount, 0.0f);
	std::vector<unsigned char> isPlastic(scount, 0);

	for(unsigned s = 0; s < scount; ++s) {
		targets.at(s) = rtarget();
		weights.at(s) = 0.5f * float(rweight());
	}

	net->addSynapses(source, targets, delays, weights, isPlastic);
	nemo_add_synapses(c_net, source, &targets[0], &delays[0], &weights[0], &isPlastic[0], targets.size());
}


void
addInhibitoryNeuron(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned nidx,
		urng_t& param)
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
	nemo_add_neuron(c_net, nidx, a, b, c, d, u, v, sigma);
}



void
addInhibitorySynapses(
		nemo::Network* net,
		nemo_network_t c_net,
		unsigned source,
		unsigned ncount,
		unsigned scount,
		uirng_t& rtarget,
		urng_t& rweight)
{
	std::vector<unsigned> targets(scount, 0);
	std::vector<unsigned> delays(scount, 1U);
	std::vector<float> weights(scount, 0.0f);
	std::vector<unsigned char> isPlastic(scount, 0);

	for(unsigned s = 0; s < scount; ++s) {
		targets.at(s) = rtarget();
		weights.at(s) = float(-rweight());
	}

	net->addSynapses(source, targets, delays, weights, isPlastic);
	nemo_add_synapses(c_net, source, &targets[0], &delays[0], &weights[0], &isPlastic[0], targets.size());
}


void
c_runSimulation(
		const nemo_network_t net,
		const nemo_configuration_t conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx)
{
	nemo_simulation_t sim = nemo_new_simulation(net, conf);

	fcycles->clear();
	fnidx->clear();

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {

		nemo_step(sim, NULL, 0);

		//! \todo could modify API here to make this nicer
		unsigned* cycles_tmp;
		unsigned* nidx_tmp;
		unsigned nfired;
		unsigned ncycles;

		nemo_read_firing(sim, &cycles_tmp, &nidx_tmp, &nfired, &ncycles);

		// push data back onto local buffers
		std::copy(cycles_tmp, cycles_tmp + nfired, back_inserter(*fcycles));
		std::copy(nidx_tmp, nidx_tmp + nfired, back_inserter(*fnidx));
	}

	nemo_delete_simulation(sim);
}


BOOST_AUTO_TEST_CASE(test_c_api)
{
	unsigned ncount = 1000;
	unsigned scount = 1000;

	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount-1));

	std::cerr << "Creating network (C++ API)\n";
	nemo::Network* net = new nemo::Network();
	std::cerr << "Creating network (C API)\n";
	nemo_network_t c_net = nemo_new_network();

	std::cerr << "Populating networks\n";
	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		if(nidx < (ncount * 4) / 5) { // excitatory
			addExcitatoryNeuron(net, c_net, nidx, randomParameter);
			addExcitatorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		} else { // inhibitory
			addInhibitoryNeuron(net, c_net, nidx, randomParameter);
			addInhibitorySynapses(net, c_net, nidx, ncount, scount, randomTarget, randomParameter);
		}
	}

	nemo::Configuration conf;

	unsigned duration = 2;

	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;

	std::cerr << "Running network (C++ API)\n";
	runSimulation(net, conf, duration, &cycles1, &nidx1);
	nemo_configuration_t c_conf = nemo_new_configuration();
	std::cerr << "Running network (C API)\n";
	c_runSimulation(c_net, c_conf, duration, &cycles2, &nidx2);
	std::cerr << "Checking results\n";
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);

	nemo_delete_configuration(c_conf);
	nemo_delete_network(c_net);
}
