#define BOOST_TEST_MODULE nemo test

#include <cmath>
#include <iostream>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include <examples.hpp>

#include "utils.hpp"


/* Run one simulation after another and make sure their firing output match
 * exactly */
void
compareSimulations(
		const nemo::Network* net1,
		const nemo::Configuration& conf1,
		const nemo::Network* net2,
		const nemo::Configuration& conf2,
		unsigned duration)
{
	std::cout << "Comparing " << conf1 << "\n"
	          << "      and " << conf2 << "\n";
	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;
	runSimulation(net1, conf1, duration, &cycles1, &nidx1);
	runSimulation(net2, conf2, duration, &cycles2, &nidx2);
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);
}


nemo::Configuration
configuration(bool stdp, unsigned partitionSize)
{
	nemo::Configuration conf;

	if(stdp) {
		std::vector<float> pre(20);
		std::vector<float> post(20);
		for(unsigned i = 0; i < 20; ++i) {
			float dt = float(i + 1);
			pre.at(i) = 1.0 * expf(-dt / 20.0f);
			pre.at(i) = -0.8 * expf(-dt / 20.0f);
		}
		conf.setStdpFunction(pre, post, 10.0, -10.0);
	}

	conf.setCudaPartitionSize(partitionSize);

	return conf;
}



void
runComparisions(nemo::Network* net)
{
	unsigned duration = 2;

	/* network should produce repeatable results both with the same partition
	 * size and with different ones. */
	{
		bool stdp_conf[2] = { false, true };
		unsigned psize_conf[3] = { 1024, 512, 256 };

		for(unsigned si=0; si < 2; ++si)
		for(unsigned pi1=0; pi1 < 3; ++pi1) 
		for(unsigned pi2=0; pi2 < 3; ++pi2) {
			nemo::Configuration conf1 = configuration(stdp_conf[si], psize_conf[pi1]);
			nemo::Configuration conf2 = configuration(stdp_conf[si], psize_conf[pi2]);
			//! \note tests with different partition sizes may fail due to
			//randomised input working differently.
			compareSimulations(net, conf1, net, conf2, duration);
		}
	}

}


/* It should be possible to create a network without any synapses */
BOOST_AUTO_TEST_CASE(simulation_without_synapses)
{
	nemo::Network net;
	net.addNeuron(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7);
	nemo::Configuration conf;
	nemo::Simulation* sim = NULL;
	BOOST_REQUIRE_NO_THROW(sim = nemo::simulation(net, conf));
	BOOST_REQUIRE_NO_THROW(sim->step());
	delete sim;
}



/* Simple ring network.
 *
 * This is useful for basic testing as the exact firing pattern is known in
 * advance. Every cycle a single neuron fires. Each neuron connected to only
 * the next neuron (in global index space) with an abnormally strong synapse,
 * so the result is the firing propagating around the ring.
 */
void
runRing(unsigned ncount, const nemo::Configuration& conf)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;

	nemo::Network net;
	for(unsigned source=0; source < ncount; ++source) {
		float v = -65.0f;
		float b = 0.2f;
		float r = 0.5f;
		float r2 = r * r;
		net.addNeuron(source, 0.02f, b, v+15.0f*r2, 8.0f-6.0f*r2, b*v, v, 0.0f);
		net.addSynapse(source, (source + 1) % ncount, 1, 1000.0f, false);
	}
	nemo::Simulation* sim = nemo::simulation(net, conf);

	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* fired;

	/* Simulate a single neuron to get the ring going */
	sim->step(std::vector<unsigned>(1,0));
	sim->flushFiringBuffer();

	for(unsigned ms=1; ms < duration; ++ms) {
		sim->step();
		sim->readFiring(&cycles, &fired);
		BOOST_CHECK_EQUAL(cycles->size(), fired->size());
		BOOST_CHECK_EQUAL(fired->size(), 1);
		BOOST_REQUIRE_EQUAL(fired->front(), ms % ncount);
	}
}


BOOST_AUTO_TEST_CASE(ring_tests)
{
	nemo::Configuration conf = configuration(false, 1024);
	runRing(1000, conf); // less than a single partition on CUDA backend
	runRing(1024, conf); // exactly one partition on CUDA backend
	runRing(2000, conf); // multiple partitions on CUDA backend
	runRing(4000, conf); // ditto
}



BOOST_AUTO_TEST_CASE(mapping_tests_random1k)
{
	// only need to create the network once
	nemo::Network* net = nemo::random1k::construct(1000, 1000);
	runComparisions(net);
	delete net;
}


BOOST_AUTO_TEST_CASE(mapping_tests_torus)
{
	//! \todo run for larger networks as well
	unsigned pcount = 1;
	unsigned m = 1000;
	unsigned sigma = 16;
	bool logging = false;

	// only need to create the network once
	nemo::Network* net = nemo::torus::construct(pcount, m, true, sigma, logging);

	runComparisions(net);
	delete net;
}



BOOST_AUTO_TEST_CASE(fixpoint_precision_specification)
{
	nemo::Network* net = nemo::random1k::construct(1000, 1000);
	nemo::Configuration conf;

	conf.setFractionalBits(26);
	conf.enableLogging();
	std::vector<unsigned> cycles, nidx, cycles2, nidx2;
	unsigned duration = 2;
	runSimulation(net, conf, duration, &cycles, &nidx);

	BOOST_REQUIRE(nidx.size() > 0);

	nemo::Configuration conf2;
	conf2.enableLogging();
	runSimulation(net, conf2, duration, &cycles2, &nidx2);
	compareSimulationResults(cycles, nidx, cycles2, nidx2);
}
