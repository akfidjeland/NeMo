#define BOOST_TEST_MODULE nemo test

#include <cmath>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>

#include <nemo.hpp>
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
			nemo::Configuration conf2 = configuration(stdp_conf[si], 1024, NEMO_BACKEND_CUDA);
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
		net.addNeuron(nidx, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
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

		for(unsigned n = 0; n < ncount; ++n) {
			if(random() < p_fire) {
				fstim.push_back(n);
			}
		}

		const std::vector<unsigned>& fired = sim->step(fstim);

		/* The neurons which just fired should be exactly the ones we just stimulated */
		sortAndCompare(fstim, fired);
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


void
testCurrentStimulus(backend_t backend)
{
	unsigned ncount = 1500;
	unsigned duration = ncount * 2;

	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Network> net(createRing(ncount));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	nemo::Simulation::current_stimulus istim;
	// add some noise before and after
	istim.push_back(std::make_pair(5U, 0.001f));
	istim.push_back(std::make_pair(8U, 0.001f));
	istim.push_back(std::make_pair(0U, 1000.0f));
	istim.push_back(std::make_pair(100U, 0.001f));
	istim.push_back(std::make_pair(1U, 0.001f));

	/* Simulate a single neuron to get the ring going */
	sim->step(istim);

	for(unsigned ms=1; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		BOOST_CHECK_EQUAL(fired.size(), 1U);
		BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
	}
}


BOOST_AUTO_TEST_SUITE(istim)

	BOOST_AUTO_TEST_CASE(cuda) {
		testCurrentStimulus(NEMO_BACKEND_CUDA);
	}

	BOOST_AUTO_TEST_CASE(cpu) {
		testCurrentStimulus(NEMO_BACKEND_CPU);
	}

BOOST_AUTO_TEST_SUITE_END()


void
runRing(unsigned ncount, nemo::Configuration conf)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;

	boost::scoped_ptr<nemo::Network> net(createRing(ncount));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	/* Simulate a single neuron to get the ring going */
	sim->step(std::vector<unsigned>(1, 0));

	for(unsigned ms=1; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		BOOST_CHECK_EQUAL(fired.size(), 1U);
		BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
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
	nemo::Network* net = nemo::random::construct(4000, 1000, false);
	runBackendComparisions(net);
	delete net;
}



BOOST_AUTO_TEST_CASE(mapping_tests_random)
{
	// only need to create the network once
	nemo::Network* net = nemo::random::construct(1000, 1000, false);
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



/* Create simulation and verify that the simulation data contains the same
 * synapses as the input network. Neurons are assumed to lie in a contigous
 * range of indices starting at n0. */
void
testGetSynapses(const nemo::Network& net,
		nemo::Configuration& conf,
		unsigned n0,
		unsigned m)
{
	unsigned fbits = 20;
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	std::vector<unsigned> ntargets;
	std::vector<unsigned> ndelays;
	std::vector<float> nweights;
	std::vector<unsigned char> nplastic;

	for(unsigned src = n0, src_end = n0 + net.neuronCount(); src < src_end; ++src) {

		std::vector<synapse_id> ids = synapseIds(src, m);

		net.getSynapses(src, ntargets, ndelays, nweights, nplastic);
		for(std::vector<float>::iterator i = nweights.begin(); i != nweights.end(); ++i) {
			*i = fx_toFloat(fx_toFix(*i, fbits), fbits);
		}

		sortAndCompare(sim->getWeights(ids), nweights);
		sortAndCompare(sim->getTargets(ids), ntargets);
		sortAndCompare(sim->getDelays(ids), ndelays);
		sortAndCompare(sim->getPlastic(ids), nplastic);
	}
}


void
testGetSynapses(backend_t backend, bool stdp)
{
	nemo::Configuration conf = configuration(stdp, 1024, backend);

	unsigned m = 1000;
	boost::scoped_ptr<nemo::Network> net1(nemo::torus::construct(4, m, stdp, 32, false));
	testGetSynapses(*net1, conf, 0, m);

	unsigned n0 = 1000000U;
	boost::scoped_ptr<nemo::Network> net2(createRing(1500, n0));
	testGetSynapses(*net2, conf, n0, 1);
}


/* The network should contain the same synapses before and after setting up the
 * simulation. The order of the synapses may differ, though. */
BOOST_AUTO_TEST_SUITE(get_synapses);

	BOOST_AUTO_TEST_CASE(cuda) {
		testGetSynapses(NEMO_BACKEND_CUDA, false);
		testGetSynapses(NEMO_BACKEND_CUDA, true);
	}

	BOOST_AUTO_TEST_CASE(cpu) {
		testGetSynapses(NEMO_BACKEND_CPU, false);
		testGetSynapses(NEMO_BACKEND_CPU, true);
	}

BOOST_AUTO_TEST_SUITE_END();


void testStdp(backend_t backend, bool noiseConnections, float reward);
void testInvalidStdpUsage(backend_t);

BOOST_AUTO_TEST_SUITE(stdp);

	BOOST_AUTO_TEST_SUITE(cuda);

		BOOST_AUTO_TEST_CASE(integral) {
			testStdp(NEMO_BACKEND_CUDA, false, 1.0);
			testStdp(NEMO_BACKEND_CUDA, true, 1.0);
		}

		BOOST_AUTO_TEST_CASE(fractional) {
			testStdp(NEMO_BACKEND_CUDA, false, 0.8);
			testStdp(NEMO_BACKEND_CUDA, true, 0.8);
		}

		BOOST_AUTO_TEST_CASE(error) {
			testInvalidStdpUsage(NEMO_BACKEND_CUDA);
		}

	BOOST_AUTO_TEST_SUITE_END();

	BOOST_AUTO_TEST_SUITE(cpu);

		BOOST_AUTO_TEST_CASE(integral) {
			testStdp(NEMO_BACKEND_CPU, false, 1.0);
			testStdp(NEMO_BACKEND_CPU, true, 1.0);
		}

		BOOST_AUTO_TEST_CASE(fractional) {
			testStdp(NEMO_BACKEND_CPU, false, 0.8);
			testStdp(NEMO_BACKEND_CPU, true, 0.8);
		}

		BOOST_AUTO_TEST_CASE(error) {
			testInvalidStdpUsage(NEMO_BACKEND_CPU);
		}

	BOOST_AUTO_TEST_SUITE_END();

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




/* create basic network with a single neuron and verify that membrane potential
 * is set correctly initially */
void
testVProbe(backend_t backend)
{
	nemo::Network net;
	float v0 = -65.0;
	net.addNeuron(0, 0.02f, 0.2f, -65.0f+15.0f*0.25f, 8.0f-6.0f*0.25f, 0.2f*-65.0f, v0, 5.0f);

	nemo::Configuration conf = configuration(false, 1024, backend);

	boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
	BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));
}



BOOST_AUTO_TEST_SUITE(vprobe)
	BOOST_AUTO_TEST_CASE(cuda) {
		testVProbe(NEMO_BACKEND_CUDA);
	}
	BOOST_AUTO_TEST_CASE(cpu) {
		testVProbe(NEMO_BACKEND_CPU);
	}
BOOST_AUTO_TEST_SUITE_END()



void
testSetNeuron(backend_t backend)
{
	float a = 0.02f;
	float b = 0.2f;
	float c = -65.0f+15.0f*0.25f;
	float d = 8.0f-6.0f*0.25f;
	float v = -65.0f;
	float u = b * v;
	float sigma = 5.0f;

	/* Create a minimal network with a single neuron */
	nemo::Network net;
	net.addNeuron(0, a, b, c, d, u, v, sigma);

	nemo::Configuration conf = configuration(false, 1024, backend);

	float v0 = 0.0f;
	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		sim->step();
		v0 = sim->getMembranePotential(0);
	}

	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		/* Marginally change the 'c' parameter. This is only used if the neuron
		 * fires (which it shouldn't do this cycle). This modification
		 * therefore should not affect the simulation result (here measured via
		 * the membrane potential) */
		sim->setNeuron(0, a, b, c+1.0f, d, u, v, sigma);
		sim->step();
		BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));
	}

	{
		/* Modify membrane potential after simulation has been created.
		 * Again the result should be the same */
		nemo::Network net1;
		net1.addNeuron(0, a, b, c, d, u, v-1.0f, sigma);
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net1, conf));
		sim->setNeuron(0, a, b, c, d, u, v, sigma);
		sim->step();
		BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));
	}
}



BOOST_AUTO_TEST_SUITE(set_neuron)
	BOOST_AUTO_TEST_CASE(cuda) {
		testSetNeuron(NEMO_BACKEND_CUDA);
	}
	BOOST_AUTO_TEST_CASE(cpu) {
		testSetNeuron(NEMO_BACKEND_CPU);
	}
BOOST_AUTO_TEST_SUITE_END()
