#define BOOST_TEST_MODULE nemo test

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

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
#include "rtest.hpp"
#include "c_api.hpp"


/* For a number of tests, we want to run both CUDA and CPU versions with the
 * same parameters. */
#ifdef NEMO_CUDA_ENABLED

// unary function
#define TEST_ALL_BACKENDS(name, fn)                                           \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU); }                       \
    BOOST_AUTO_TEST_CASE(cuda) { fn(NEMO_BACKEND_CUDA); }                     \
    BOOST_AUTO_TEST_SUITE_END()

// n-ary function for n >= 2
#define TEST_ALL_BACKENDS_N(name, fn,...)                                     \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU, __VA_ARGS__); }          \
    BOOST_AUTO_TEST_CASE(cuda) { fn(NEMO_BACKEND_CUDA, __VA_ARGS__); }        \
    BOOST_AUTO_TEST_SUITE_END()

#else

#define TEST_ALL_BACKENDS(name, fn)                                           \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU); }                       \
    BOOST_AUTO_TEST_SUITE_END()

#define TEST_ALL_BACKENDS_N(name, fn,...)                                     \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU, __VA_ARGS__); }          \
    BOOST_AUTO_TEST_SUITE_END()

#endif

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
	TEST_ALL_BACKENDS(no_outgoing, no_outgoing::run)
	TEST_ALL_BACKENDS(invalid_targets, invalid_targets::run)
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


TEST_ALL_BACKENDS(fstim, testFiringStimulus)


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


TEST_ALL_BACKENDS(istim, testCurrentStimulus)


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


BOOST_AUTO_TEST_SUITE(ring_tests)
	nemo::Configuration conf = configuration(false, 1024);
	BOOST_AUTO_TEST_CASE(n1000) {
		runRing(1000, conf); // less than a single partition on CUDA backend
	}
	BOOST_AUTO_TEST_CASE(n1024) {
		runRing(1024, conf); // exactly one partition on CUDA backend
	}
	BOOST_AUTO_TEST_CASE(n2000) {
		runRing(2000, conf); // multiple partitions on CUDA backend
	}
	BOOST_AUTO_TEST_CASE(n4000) {
		runRing(4000, conf); // ditto
	}
BOOST_AUTO_TEST_SUITE_END()


#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_CASE(compare_backends)
{
	nemo::Network* net = nemo::random::construct(4000, 1000, false);
	runBackendComparisions(net);
	delete net;
}
#endif



#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_CASE(mapping_tests_random)
{
	// only need to create the network once
	boost::scoped_ptr<nemo::Network> net(nemo::random::construct(1000, 1000, true));
	runComparisions(net.get());
}
#endif



#ifdef NEMO_CUDA_ENABLED
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
#endif


void
testNonContigousNeuronIndices(backend_t backend, unsigned n0, unsigned nstep)
{
	unsigned ncount = 1000;
	bool stdp = false;

	boost::scoped_ptr<nemo::Network> net0(createRing(ncount, 0, false, nstep));
	boost::scoped_ptr<nemo::Network> net1(createRing(ncount, n0, false, nstep));

	std::vector<unsigned> cycles0, cycles1;
	std::vector<unsigned> fired0, fired1;

	unsigned seconds = 2;
	nemo::Configuration conf = configuration(false, 1024, backend);

	runSimulation(net0.get(), conf, seconds, &cycles0, &fired0, stdp, std::vector<unsigned>(1, 0));
	runSimulation(net1.get(), conf, seconds, &cycles1, &fired1, stdp, std::vector<unsigned>(1, n0));

	/* The results should be the same, except firing indices
	 * should have the same offset. */
	BOOST_REQUIRE_EQUAL(cycles0.size(), cycles1.size());
	BOOST_REQUIRE_EQUAL(fired0.size(), seconds*ncount);
	BOOST_REQUIRE_EQUAL(fired1.size(), seconds*ncount);

	for(unsigned i = 0; i < cycles0.size(); ++i) {
		BOOST_REQUIRE_EQUAL(cycles0.at(i), cycles1.at(i));
		BOOST_REQUIRE_EQUAL(fired0.at(i), fired1.at(i) - n0);
	}

	//! \todo also add ring networks with different steps.
}


BOOST_AUTO_TEST_SUITE(non_contigous_indices)
	TEST_ALL_BACKENDS_N(contigous_low, testNonContigousNeuronIndices, 1, 1)
	TEST_ALL_BACKENDS_N(contigous_high, testNonContigousNeuronIndices, 1000000, 1)
	TEST_ALL_BACKENDS_N(non_contigous_low, testNonContigousNeuronIndices, 1, 4)
	TEST_ALL_BACKENDS_N(non_contigous_high, testNonContigousNeuronIndices, 1000000, 4)
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

	for(unsigned src = n0, src_end = n0 + net.neuronCount(); src < src_end; ++src) {

		const std::vector<synapse_id>& ids = sim->getSynapsesFrom(src);

		for(std::vector<synapse_id>::const_iterator i = ids.begin(); i != ids.end(); ++i) {
			BOOST_REQUIRE_EQUAL(sim->getSynapseTarget(*i), net.getSynapseTarget(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapseDelay(*i), net.getSynapseDelay(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapsePlastic(*i), net.getSynapsePlastic(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapsePlastic(*i),
					fx_toFloat(fx_toFix(net.getSynapsePlastic(*i), fbits), fbits));

		}
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


void
testWriteOnlySynapses(backend_t backend)
{
	bool stdp = false;
	nemo::Configuration conf = configuration(stdp, 1024, backend);
	conf.setWriteOnlySynapses();
	boost::scoped_ptr<nemo::Network> net(createRing(10, 0, true));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
	sim->step();
	BOOST_REQUIRE_THROW(const std::vector<synapse_id> ids = sim->getSynapsesFrom(0), nemo::exception);

	/* so, we cant use getSynapsesFrom, but create a synapse id anyway */
	nidx_t neuron = 1;
	unsigned synapse = 0;
	synapse_id id = (uint64_t(neuron) << 32) | uint64_t(synapse);

	BOOST_REQUIRE_THROW(sim->getSynapseWeight(id), nemo::exception);
}



void
testGetSynapsesFromUnconnectedNeuron(backend_t backend)
{
	nemo::Network net;
	for(int nidx = 0; nidx < 4; ++nidx) {
		net.addNeuron(nidx, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
	}
	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));
	sim->step();

	/* If neuron is invalid we should throw */
	BOOST_REQUIRE_THROW(sim->getSynapsesFrom(4), nemo::exception);

	/* However, if neuron is unconnected, we should get an empty list */
	std::vector<synapse_id> ids;
	BOOST_REQUIRE_NO_THROW(ids = sim->getSynapsesFrom(3));
	BOOST_REQUIRE(ids.size() == 0);
}


/* The network should contain the same synapses before and after setting up the
 * simulation. The order of the synapses may differ, though. */
BOOST_AUTO_TEST_SUITE(get_synapses);
	TEST_ALL_BACKENDS_N(nostdp, testGetSynapses, false)
	TEST_ALL_BACKENDS_N(stdp, testGetSynapses, true)
	TEST_ALL_BACKENDS(write_only, testWriteOnlySynapses)
	TEST_ALL_BACKENDS(from_unconnected, testGetSynapsesFromUnconnectedNeuron)
BOOST_AUTO_TEST_SUITE_END();


void testStdp(backend_t backend, bool noiseConnections, float reward);
void testInvalidStdpUsage(backend_t);


void
testStdpWithAllStatic(backend_t backend)
{
	boost::scoped_ptr<nemo::Network> net(nemo::random::construct(1000, 1000, false));
	nemo::Configuration conf = configuration(true, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
	for(unsigned s=0; s<4; ++s) {
		for(unsigned ms=0; ms<1000; ++ms) {
			sim->step();
		}
		sim->applyStdp(1.0);
	}
}



BOOST_AUTO_TEST_SUITE(stdp);
	TEST_ALL_BACKENDS_N(simple, testStdp, false, 1.0)
	TEST_ALL_BACKENDS_N(noisy, testStdp, true, 1.0)
	TEST_ALL_BACKENDS_N(simple_fractional_reward, testStdp, false, 0.8)
	TEST_ALL_BACKENDS_N(noise_fractional_reward, testStdp, true, 0.9)
	TEST_ALL_BACKENDS(invalid, testInvalidStdpUsage)
	TEST_ALL_BACKENDS(all_static, testStdpWithAllStatic)
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
#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_SUITE(high_firing)
	BOOST_AUTO_TEST_CASE(cuda) {
		testHighFiring(NEMO_BACKEND_CUDA, false);
	}
BOOST_AUTO_TEST_SUITE_END()
#endif




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


TEST_ALL_BACKENDS(vprobe, testVProbe)



/* Both the simulation and network classes have neuron setters. Here we perform
 * the same test for both. */
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

	/* setNeuron should only succeed for existing neurons */
	BOOST_REQUIRE_THROW(net.setNeuron(0, a, b, c, d, u, v, sigma), nemo::exception);

	net.addNeuron(0, a, b, c-0.1, d, u, v-1.0, sigma);

	/* Invalid neuron */
	BOOST_REQUIRE_THROW(net.getNeuronParameter(1, 0), nemo::exception);
	BOOST_REQUIRE_THROW(net.getNeuronState(1, 0), nemo::exception);

	/* Invalid parameter */
	BOOST_REQUIRE_THROW(net.getNeuronParameter(0, 5), nemo::exception);
	BOOST_REQUIRE_THROW(net.getNeuronState(0, 2), nemo::exception);

	float e = 0.1;
	BOOST_REQUIRE_NO_THROW(net.setNeuron(0, a-e, b-e, c-e, d-e, u-e, v-e, sigma-e));
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 0), a-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 1), b-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 2), c-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 3), d-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 4), sigma-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 0), u-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 1), v-e);

	/* Try setting individual parameters during construction */

	net.setNeuronParameter(0, 0, a);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 0), a);

	net.setNeuronParameter(0, 1, b);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 1), b);

	net.setNeuronParameter(0, 2, c);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 2), c);

	net.setNeuronParameter(0, 3, d);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 3), d);

	net.setNeuronParameter(0, 4, sigma);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 4), sigma);

	net.setNeuronState(0, 0, u);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 0), u);

	net.setNeuronState(0, 1, v);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 1), v);

	/* Invalid neuron */
	BOOST_REQUIRE_THROW(net.setNeuronParameter(1, 0, 0.0f), nemo::exception);
	BOOST_REQUIRE_THROW(net.setNeuronState(1, 0, 0.0f), nemo::exception);

	/* Invalid parameter */
	BOOST_REQUIRE_THROW(net.setNeuronParameter(0, 5, 0.0f), nemo::exception);
	BOOST_REQUIRE_THROW(net.setNeuronState(0, 2, 0.0f), nemo::exception);

	nemo::Configuration conf = configuration(false, 1024, backend);

	/* Try setting individual parameters during simulation */
	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));

		sim->step();

		sim->setNeuronState(0, 0, u-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 0), u-e);

		sim->setNeuronState(0, 1, v-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 1), v-e);

		/* Get the data back to later verify that it does in fact change during
		 * simulation, rather than being overwritten again on subsequent
		 * simulation steps */
		float u0 = sim->getNeuronState(0, 0);
		float v0 = sim->getNeuronState(0, 1);

		sim->setNeuronParameter(0, 0, a-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a-e);

		sim->setNeuronParameter(0, 1, b-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b-e);

		sim->setNeuronParameter(0, 2, c-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c-e);

		sim->setNeuronParameter(0, 3, d-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d-e);

		sim->setNeuronParameter(0, 4, sigma-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma-e);

		sim->step();

		/* After simulating one more step all the parameter should remain the
		 * same, whereas all the state variables should have changed */
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma-e);

		BOOST_REQUIRE(sim->getNeuronState(0, 0) != u0);
		BOOST_REQUIRE(sim->getNeuronState(0, 1) != v0);


		/* Invalid neuron */
		BOOST_REQUIRE_THROW(sim->setNeuronParameter(1, 0, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->setNeuronState(1, 0, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronParameter(1, 0), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronState(1, 0), nemo::exception);

		/* Invalid parameter */
		BOOST_REQUIRE_THROW(sim->setNeuronParameter(0, 5, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->setNeuronState(0, 2, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronParameter(0, 5), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronState(0, 2), nemo::exception);
	}

	float v0 = 0.0f;
	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		sim->step();
		v0 = sim->getMembranePotential(0);
	}

	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 0), u);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 1), v);
		/* Marginally change the 'c' parameter. This is only used if the neuron
		 * fires (which it shouldn't do this cycle). This modification
		 * therefore should not affect the simulation result (here measured via
		 * the membrane potential) */
		sim->setNeuron(0, a, b, c+1.0f, d, u, v, sigma);

		sim->step();

		BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));

		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c+1.0f);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma);
	}

	{
		/* Ensure that when setting the state variable, it is not copied
		 * multiple times */
		nemo::Network net0;
		net0.addNeuron(0, a, b, c, d, u, v, 0.0f);

		boost::scoped_ptr<nemo::Simulation> sim0(simulation(net0, conf));
		sim0->step();
		sim0->step();
		float v0 = sim0->getMembranePotential(0);

		boost::scoped_ptr<nemo::Simulation> sim1(simulation(net0, conf));
		sim1->step();
		sim1->setNeuron(0, a, b, c, d, u, v, 0.0f);
		sim1->step();
		sim1->step();
		float v1 = sim1->getMembranePotential(0);

		BOOST_REQUIRE_EQUAL(v0, v1);
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



TEST_ALL_BACKENDS(set_neuron, testSetNeuron)

BOOST_AUTO_TEST_SUITE(regression)
	BOOST_AUTO_TEST_CASE(torus) {
		runTorus(false);
	}
BOOST_AUTO_TEST_SUITE_END()



BOOST_AUTO_TEST_SUITE(c_api)

	BOOST_AUTO_TEST_SUITE(comparison)
		BOOST_AUTO_TEST_CASE(nostim) { nemo::test::c_api::compareWithCpp(false, false); }
		BOOST_AUTO_TEST_CASE(fstim) { nemo::test::c_api::compareWithCpp(true, false); }
		BOOST_AUTO_TEST_CASE(istim) { nemo::test::c_api::compareWithCpp(false, true); }
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_CASE(synapse_ids) { nemo::test::c_api::testSynapseId(); }
	BOOST_AUTO_TEST_CASE(set_neuron) { nemo::test::c_api::testSetNeuron(); }

	BOOST_AUTO_TEST_SUITE(get_synapse)
		TEST_ALL_BACKENDS_N(n0, nemo::test::c_api::testGetSynapses, 0)
		TEST_ALL_BACKENDS_N(n1000, nemo::test::c_api::testGetSynapses, 1000)
	BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()

