#define BOOST_TEST_MODULE nemo test

#include <cmath>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <nemo/constants.h>
#include <nemo/fixedpoint.hpp>
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
configuration(bool stdp, unsigned partitionSize,
		backend_t backend = NEMO_BACKEND_CUDA)
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
	conf.setBackend(backend);

	return conf;
}



void
runComparisions(nemo::Network* net)
{
	unsigned duration = 2;

	/* simulations should produce repeatable results both with the same
	 * partition size and with different ones. */
	{
		bool stdp_conf[2] = { false, true };
		unsigned psize_conf[3] = { 1024, 512, 256 };

		for(unsigned si=0; si < 2; ++si)
		for(unsigned pi1=0; pi1 < 3; ++pi1) 
		for(unsigned pi2=0; pi2 < 3; ++pi2) {
			nemo::Configuration conf1 = configuration(stdp_conf[si], psize_conf[pi1]);
			nemo::Configuration conf2 = configuration(stdp_conf[si], psize_conf[pi2]);
			compareSimulations(net, conf1, net, conf2, duration);
		}
	}

}


void
runBackendComparisions(nemo::Network* net)
{
	unsigned duration = 2; // seconds

	/* simulations should produce repeatable results regardless of the backend
	 * which is used */
	//! \todo add test for stdp as well;
	{
		bool stdp_conf[1] = { false };

		for(unsigned si=0; si < 1; ++si) {
			nemo::Configuration conf1 = configuration(stdp_conf[si], 1024, NEMO_BACKEND_CPU);
			nemo::Configuration conf2 = configuration(stdp_conf[si], 1024, NEMO_BACKEND_CUDA);
			compareSimulations(net, conf1, net, conf2, duration);
		}
	}

}



void
runSimple(unsigned startNeuron, unsigned neuronCount)
{
	nemo::Network net;
	for(int nidx = 0; nidx < 4; ++nidx) {
		net.addNeuron(nidx, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7);
	}
	nemo::Configuration conf;
	nemo::Simulation* sim = NULL;
	BOOST_REQUIRE_NO_THROW(sim = nemo::simulation(net, conf));
	BOOST_REQUIRE_NO_THROW(sim->step());
	delete sim;
}



BOOST_AUTO_TEST_CASE(simulation_unary_network)
{
	runSimple(0, 1);
}



/* It should be possible to create a network without any synapses */
BOOST_AUTO_TEST_CASE(simulation_without_synapses)
{
	runSimple(0, 4);
}


/* We should be able to deal with networs with neuron indices not starting at
 * zero */
BOOST_AUTO_TEST_CASE(simulation_one_based_indices)
{
	runSimple(1, 4);
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


BOOST_AUTO_TEST_CASE(compare_backends)
{
	nemo::Network* net = nemo::random1k::construct(4000, 1000);
	runBackendComparisions(net);
	delete net;
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



//! \todo test this for cpu backend as well
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
	BOOST_REQUIRE(nidx2.size() > 0);
	compareSimulationResults(cycles, nidx, cycles2, nidx2);
}



template<typename T>
void
// pass by value here since the simulation data cannot be modified
sortAndCompare(std::vector<T> a, std::vector<T> b)
{
	std::sort(a.begin(), a.end());
	std::sort(b.begin(), b.end());
	BOOST_REQUIRE(a.size() == b.size());
	for(size_t i = 0; i < a.size(); ++i) {
		BOOST_REQUIRE(a[i] == b[i]);
	}
}



void
testGetSynapses(backend_t backend)
{
	boost::scoped_ptr<nemo::Network> net(nemo::random1k::construct(4000, 1000));

	nemo::Configuration conf;
	conf.setBackend(backend);
	unsigned fbits = 22;
	conf.setFractionalBits(fbits);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	std::vector<unsigned> ntargets;
	std::vector<unsigned> ndelays;
	std::vector<float> nweights;
	std::vector<unsigned char> nplastic;
	const std::vector<unsigned> *stargets;
	const std::vector<unsigned> *sdelays;
	const std::vector<float> *sweights;
	const std::vector<unsigned char> *splastic;

	for(unsigned src = 0, src_end = net->neuronCount(); src < src_end; ++src) {
		net->getSynapses(src, ntargets, ndelays, nweights, nplastic);
		sim->getSynapses(src, &stargets, &sdelays, &sweights, &splastic);
		sortAndCompare(ntargets, *const_cast<std::vector<unsigned>*>(stargets));
		for(std::vector<float>::iterator i = nweights.begin(); i != nweights.end(); ++i) {
			*i = fx_toFloat(fx_toFix(*i, fbits), fbits);
		}
		sortAndCompare(nweights, *const_cast<std::vector<float>*>(sweights));
		sortAndCompare(ndelays, *const_cast<std::vector<unsigned>*>(sdelays));
		sortAndCompare(nplastic, *const_cast<std::vector<unsigned char>*>(splastic));
	}
}



/* The network should contain the same synapses before and after setting up the
 * simulation. The order of the synapses may differ, though. */
BOOST_AUTO_TEST_SUITE(get_synapses);

	BOOST_AUTO_TEST_CASE(cuda) {
		testGetSynapses(NEMO_BACKEND_CUDA);
	}

	BOOST_AUTO_TEST_CASE(cpu) {
		testGetSynapses(NEMO_BACKEND_CPU);
	}

BOOST_AUTO_TEST_SUITE_END();
