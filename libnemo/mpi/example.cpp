#include <iostream>
#include <fstream>
#include <iterator>

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

	if(argc < 3) {
		std::cerr << "usage: example <ncount> <outfile>\n";
		return -1;
	}

	unsigned ncount = atoi(argv[1]);
	unsigned scount = 1000;
	char* filename = argv[2];

	try {
		if(world.rank() == nemo::mpi::MASTER) {

			nemo::Network* net = nemo::random1k::construct(ncount, scount);
			nemo::Configuration conf;
			nemo::mpi::Master sim(env, world, *net, conf);

			std::ofstream file(filename);

			//! \todo run this for longer
			for(unsigned ms=0; ms < 100; ++ms) {
				sim.step();
				const std::vector<unsigned>& firing = sim.readFiring();
				file << ms << ": ";
				std::copy(firing.begin(), firing.end(), std::ostream_iterator<unsigned>(file, " "));
				file << std::endl;
			}
			delete net;
		} else {
			nemo::mpi::Worker sim(env, world);
		}
	} catch (nemo::exception& e) {
		std::cerr << world.rank() << ":" << e.what() << std::endl;
		env.abort(e.errno());
	} catch (boost::mpi::exception& e) {
		std::cerr << world.rank() << ": " << e.what() << std::endl;
		env.abort(-1);
	}

	return 0;
}
