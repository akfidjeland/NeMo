#define BOOST_TEST_MODULE nemo test

#include <cmath>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>

#include <nemo.hpp>
#include <nemo/constants.h>
#include <nemo/fixedpoint.hpp>
#include <examples.hpp>

#include "test.hpp"
#include "utils.hpp"


typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;


/* Run one simulation after another and make sure their firing output match
 * exactly */
void
compareSimulations(
		const nemo::Network* net1,
		const nemo::Configuration& conf1,
		const nemo::Network* net2,
		const nemo::Configuration& conf2,
		unsigned duration,
		bool stdp)
{
	std::cout << "Comparing " << conf1 << "\n"
	          << "      and " << conf2 << "\n";
	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;
	runSimulation(net1, conf1, duration, &cycles1, &nidx1, stdp);
	runSimulation(net2, conf2, duration, &cycles2, &nidx2, stdp);
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);
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
			compareSimulations(net, conf1, net, conf2, duration, stdp_conf[si]);
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
			conf1.setFractionalBits(26);
			nemo::Configuration conf2 = configuration(stdp_conf[si], 1024, NEMO_BACKEND_CUDA);
			conf2.setFractionalBits(26);
			compareSimulations(net, conf1, net, conf2, duration, stdp_conf[si]);
		}
	}

}



//! \todo migrate to networks.cpp
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

BOOST_AUTO_TEST_SUITE(networks)

	BOOST_AUTO_TEST_SUITE(no_outgoing)

		BOOST_AUTO_TEST_CASE(cpu) {
			no_outgoing::run(NEMO_BACKEND_CPU);
		}

		BOOST_AUTO_TEST_CASE(cuda) {
			no_outgoing::run(NEMO_BACKEND_CUDA);
		}

	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_SUITE(invalid_targets)

		BOOST_AUTO_TEST_CASE(cpu) {
			invalid_targets::run(NEMO_BACKEND_CPU);
		}

		BOOST_AUTO_TEST_CASE(cuda) {
			invalid_targets::run(NEMO_BACKEND_CUDA);
		}

	BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()


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


void
testFiringStimulus(backend_t backend)
{
	unsigned ncount = 3000; // make sure to cross partition boundaries
	unsigned cycles = 1000;
	unsigned firing = 10;   // every cycle
	double p_fire = double(firing) / double(ncount);

	nemo::Network net;
	for(unsigned nidx = 0; nidx < ncount; ++nidx) {
		addExcitatoryNeuron(nidx, net);
	}

	nemo::Configuration conf;
	setBackend(backend, conf);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	rng_t rng;
	urng_t random(rng, boost::uniform_real<double>(0, 1));

	for(unsigned t = 0; t < cycles; ++t) {
		std::vector<unsigned> fstim;
		const std::vector<unsigned>* fired;
		const std::vector<unsigned>* cycles;

		for(unsigned n = 0; n < ncount; ++n) {
			if(random() < p_fire) {
				fstim.push_back(n);
			}
		}

		sim->step(fstim);
		sim->readFiring(&cycles, &fired);

		/* The neurons which just fired should be exactly the ones we just stimulated */
		sortAndCompare(fstim, *fired);
	}
}


BOOST_AUTO_TEST_SUITE(fstim)

	BOOST_AUTO_TEST_CASE(cuda) {
		testFiringStimulus(NEMO_BACKEND_CUDA);
	}

	BOOST_AUTO_TEST_CASE(cpu) {
		testFiringStimulus(NEMO_BACKEND_CPU);
	}

BOOST_AUTO_TEST_SUITE_END()




/* Simple ring network.
 *
 * This is useful for basic testing as the exact firing pattern is known in
 * advance. Every cycle a single neuron fires. Each neuron connected to only
 * the next neuron (in global index space) with an abnormally strong synapse,
 * so the result is the firing propagating around the ring.
 */
nemo::Network*
createRing(unsigned ncount, unsigned n0 = 0)
{
	nemo::Network* net = new nemo::Network;
	for(unsigned source=n0; source < n0 + ncount; ++source) {
		float v = -65.0f;
		float b = 0.2f;
		float r = 0.5f;
		float r2 = r * r;
		net->addNeuron(source, 0.02f, b, v+15.0f*r2, 8.0f-6.0f*r2, b*v, v, 0.0f);
		net->addSynapse(source, n0 + ((source - n0 + 1) % ncount), 1, 1000.0f, false);
	}
	return net;
}


void
runRing(unsigned ncount, nemo::Configuration conf)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;

	boost::scoped_ptr<nemo::Network> net(createRing(ncount));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

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
	runSimulation(net, conf, duration, &cycles, &nidx, false);

	BOOST_REQUIRE(nidx.size() > 0);

	nemo::Configuration conf2;
	conf2.enableLogging();
	runSimulation(net, conf2, duration, &cycles2, &nidx2, false);
	BOOST_REQUIRE(nidx2.size() > 0);
	compareSimulationResults(cycles, nidx, cycles2, nidx2);
}


void
testNonContigousNeuronIndices(backend_t backend, unsigned n0)
{
	unsigned ncount = 1000;
	bool stdp = false;

	boost::scoped_ptr<nemo::Network> net0(createRing(ncount, 0));
	boost::scoped_ptr<nemo::Network> net1(createRing(ncount, n0));

	std::vector<unsigned> cycles0, cycles1;
	std::vector<unsigned> fired0, fired1;

	unsigned seconds = 2;
	nemo::Configuration conf = configuration(false, 1024, backend);
	conf.setFractionalBits(16);

	runSimulation(net0.get(), conf, seconds, &cycles0, &fired0, stdp, std::vector<unsigned>(1, 0));
	runSimulation(net1.get(), conf, seconds, &cycles1, &fired1, stdp, std::vector<unsigned>(1, n0));

	/* The results should be the same, except firing indices
	 * should have the same offset. */
	BOOST_REQUIRE_EQUAL(cycles0.size(), cycles1.size());
	BOOST_REQUIRE_EQUAL(fired0.size(), fired1.size());

	for(unsigned i = 0; i < cycles0.size(); ++i) {
		BOOST_REQUIRE_EQUAL(cycles0.at(i), cycles1.at(i));
		BOOST_REQUIRE_EQUAL(fired0.at(i), fired1.at(i) - n0);
	}

	//! \todo also add ring networks with different steps.
}


BOOST_AUTO_TEST_SUITE(non_contigous_indices)

	BOOST_AUTO_TEST_CASE(cpu) {
		testNonContigousNeuronIndices(NEMO_BACKEND_CPU, 1);
		testNonContigousNeuronIndices(NEMO_BACKEND_CPU, 1000000);
	}

	BOOST_AUTO_TEST_CASE(cuda) {
		testNonContigousNeuronIndices(NEMO_BACKEND_CUDA, 1);
		testNonContigousNeuronIndices(NEMO_BACKEND_CUDA, 1000000);
	}

BOOST_AUTO_TEST_SUITE_END()



void
testGetSynapses(backend_t backend)
{
	boost::scoped_ptr<nemo::Network> net(nemo::random1k::construct(4000, 1000));

	nemo::Configuration conf;
	setBackend(backend, conf);
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


void testStdp(backend_t backend, bool noiseConnections);

BOOST_AUTO_TEST_SUITE(stdp);
	BOOST_AUTO_TEST_CASE(cuda) {
		//testStdp(NEMO_BACKEND_CUDA, false);
		testStdp(NEMO_BACKEND_CUDA, true);
	}
	//! \todo add test for CPU as well
BOOST_AUTO_TEST_SUITE_END();


void
testHighFiring(backend_t backend, bool stdp)
{
	//! \todo run for larger networks as well
	unsigned pcount = 1;
	unsigned m = 1000;
	unsigned sigma = 16;
	bool logging = false;


	boost::scoped_ptr<nemo::Network> net(nemo::torus::construct(pcount, m, stdp, sigma, logging));
	nemo::Configuration conf = configuration(stdp, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	std::vector<unsigned> fstim;
	for(unsigned n = 0; n < net->neuronCount(); ++n) {
		fstim.push_back(n);
	}

	for(unsigned ms = 0; ms < 1000; ++ms) {
		BOOST_REQUIRE_NO_THROW(sim->step(fstim));
	}
}


/* The firing queue should be able to handle all firing rates. It might be best
 * to enable device assertions for this test.  */
BOOST_AUTO_TEST_SUITE(high_firing)
	BOOST_AUTO_TEST_CASE(cuda) {
		testHighFiring(NEMO_BACKEND_CUDA, false);
	}
BOOST_AUTO_TEST_SUITE_END()

