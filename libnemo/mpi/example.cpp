#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <nemo.hpp>
#include <exception.hpp>
#include <examples.hpp>
#include "nemo_mpi.hpp"


int
main(int argc, char* argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	try {
		if(world.rank() == nemo::mpi::MASTER) {
			//! \todo get neuron count from command-line
			nemo::Network* net = nemo::random1k::construct(2048, 1024);
			nemo::Configuration conf;
			nemo::mpi::Master sim(world, *net, conf);
			sim.step();
			sim.step();
			sim.step();
			delete net;
		} else {
			nemo::mpi::Worker sim(world);
		}
	} catch (nemo::exception& e) {
		std::cerr << e.what() << std::endl;
		env.abort(e.errno());
	} catch (boost::mpi::exception& e) {
		std::cerr << e.what() << std::endl;
		env.abort(-1);
	}

	return 0;
}
