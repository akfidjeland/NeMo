#define BOOST_TEST_MODULE nemo test
#include <boost/test/unit_test.hpp>
#include <nemo.hpp>

#define TORUS_NO_MAIN
#include "torus.cpp"


/* Run simulation for given length and return result in output vector */
void
runSimulation(
		const nemo::Network* net,
		const nemo::Configuration& conf,
		unsigned seconds,
		std::vector<unsigned> fcycles,
		std::vector<unsigned> fnidx)
{
	nemo::Simulation* sim = nemo::Simulation::create(*net, conf);

	fcycles.clear();
	fnidx.clear();

	//! todo vary the step size between reads to firing buffer
	
	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {
		sim->stepSimulation();

		//! \todo could modify API here to make this nicer
		const std::vector<unsigned>* cycles_tmp;
		const std::vector<unsigned>* nidx_tmp;

		sim->readFiring(&cycles_tmp, &nidx_tmp);

		// push data back onto local buffers
		std::copy(cycles_tmp->begin(), cycles_tmp->end(), back_inserter(fcycles));
		std::copy(nidx_tmp->begin(), nidx_tmp->end(), back_inserter(fnidx));
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
	runSimulation(net1, conf1, duration, cycles1, nidx1);
	runSimulation(net2, conf2, duration, cycles2, nidx2);

	BOOST_REQUIRE(cycles1.size() == nidx1.size());
	BOOST_REQUIRE(cycles2.size() == nidx2.size());
	BOOST_REQUIRE(cycles1.size() == cycles2.size());

	for(size_t i = 0; i < cycles1.size(); ++i) {
		// no point continuing after first divergence, it's only going to make
		// output hard to read.
		BOOST_CHECK(cycles1.at(i) == cycles2.at(i) + 1);
		BOOST_REQUIRE(nidx1.at(i) == nidx2.at(i));
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



BOOST_AUTO_TEST_CASE(mapping_tests)
{
	unsigned pcount = 1;
	unsigned m = 1000;
	bool stdp = false;
	unsigned sigma = 16;
	const bool logging = false;
	unsigned duration = 2;

	// only need to create the network once
	nemo::Network* net = construct(pcount, m, stdp, sigma, logging);
	nemo::Configuration conf = configure(stdp, logging);

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
			compareSimulations(net, conf1, net, conf2, 2);
		}
	}

	delete net;
}
