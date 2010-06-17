#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nemo test

#include <iostream>
#include <cmath>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include <examples.hpp>


/* Run simulation for given length and return result in output vector */
void
runSimulation(
		const nemo::Network* net,
		const nemo::Configuration& conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx)
{
	nemo::Simulation* sim = nemo::Simulation::create(*net, conf);

	fcycles->clear();
	fnidx->clear();

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {
		sim->step();

		//! \todo could modify API here to make this nicer
		const std::vector<unsigned>* cycles_tmp;
		const std::vector<unsigned>* nidx_tmp;

		sim->readFiring(&cycles_tmp, &nidx_tmp);

		// push data back onto local buffers
		std::copy(cycles_tmp->begin(), cycles_tmp->end(), back_inserter(*fcycles));
		std::copy(nidx_tmp->begin(), nidx_tmp->end(), back_inserter(*fnidx));
	}

	delete sim;
}



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

	BOOST_CHECK_EQUAL(cycles1.size(), nidx1.size());
	BOOST_CHECK_EQUAL(cycles2.size(), nidx2.size());
	BOOST_CHECK_EQUAL(cycles1.size(), cycles2.size());

	for(size_t i = 0; i < cycles1.size(); ++i) {
		// no point continuing after first divergence, it's only going to make
		// output hard to read.
		BOOST_CHECK_EQUAL(cycles1.at(i), cycles2.at(i));
		BOOST_CHECK_EQUAL(nidx1.at(i), nidx2.at(i));
		if(nidx1.at(i) != nidx2.at(i)) {
			BOOST_FAIL("c" << cycles1.at(i) << "/" << cycles2.at(i));
		}
	}
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
	bool stdp = false;
	const bool logging = false;
	unsigned duration = 2;
	nemo::Configuration conf = configuration(stdp, logging);

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
			compareSimulations(net, conf1, net, conf2, 2);
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
	BOOST_REQUIRE_NO_THROW(sim = nemo::Simulation::create(net, conf));
	BOOST_REQUIRE_NO_THROW(sim->step());
	delete sim;
}


BOOST_AUTO_TEST_CASE(mapping_tests_random1k)
{
	unsigned m = 1000;
	unsigned sigma = 16;

	// only need to create the network once
	nemo::Network* net = nemo::random1k::construct(1000);
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
