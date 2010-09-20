#define BOOST_TEST_MODULE nemo_mpi test

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo_mpi.hpp>
#include <nemo.hpp>


/* ! \note if using this code elsewhere, factor out. It's used
 * in test.cpp as well. */
void
runRing(unsigned ncount, 
		boost::mpi::environment& env,
		boost::mpi::communicator& world)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;
	// const unsigned duration = ncount * 3 / 2;

	nemo::Network net;
	for(unsigned source=0; source < ncount; ++source) {
		float v = -65.0f;
		float b = 0.2f;
		float r = 0.5f;
		float r2 = r * r;
		net.addNeuron(source, 0.02f, b, v+15.0f*r2, 8.0f-6.0f*r2, b*v, v, 0.0f);
		net.addSynapse(source, (source + 1) % ncount, 1, 1000.0f, false);
	}

	nemo::Configuration conf;
	conf.disableLogging();
	conf.setCudaPartitionSize(1024);

	nemo::mpi::Master sim(env, world, net, conf);

	/* Simulate a single neuron to get the ring going */
	sim.step(std::vector<unsigned>(1, 0));

	sim.readFiring();

	for(unsigned ms=1; ms < duration; ++ms) {
		sim.step();
		const std::vector<unsigned>& fired = sim.readFiring();
		BOOST_REQUIRE_EQUAL(fired.size(), 1U);
		BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
	}
}



void
ring_mpi(boost::mpi::environment& env,
		boost::mpi::communicator& world,
		unsigned ncount)
{
	if(world.rank() == nemo::mpi::MASTER) {
		runRing(ncount, env, world);
	} else {
		nemo::mpi::runWorker(env, world);
	}
}



BOOST_AUTO_TEST_CASE(ring_tests)
{
	boost::mpi::environment env;
	boost::mpi::communicator world;

	ring_mpi(env, world, 512);  // less than a single partition on CUDA backend
	ring_mpi(env, world, 1024); // exactly one partition on CUDA backend
	ring_mpi(env, world, 2000); // multiple partitions on CUDA backend
}
