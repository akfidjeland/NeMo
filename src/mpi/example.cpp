#include <iostream>
#include <fstream>
#include <iterator>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <nemo/exception.hpp>
#include <examples.hpp>

#include "nemo_mpi.hpp"


int
run(int argc, char* argv[],
		unsigned ncount, unsigned scount, unsigned duration,
		const char* filename)
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	try {
		if(world.rank() == nemo::mpi::MASTER) {

			boost::scoped_ptr<nemo::Network> net(nemo::random::construct(ncount, scount, false));
			nemo::Configuration conf;
			nemo::mpi::Master sim(env, world, *net, conf);

			std::ofstream file(filename);

			sim.resetTimer();
			for(unsigned ms=0; ms < duration; ++ms) {
				sim.step();
				const std::vector<unsigned>& firing = sim.readFiring();
				file << ms << ": ";
				std::copy(firing.begin(), firing.end(), std::ostream_iterator<unsigned>(file, " "));
				file << std::endl;
			}

			std::cout << "Simulated " << sim.elapsedSimulation() << "ms "
				<< "in " << sim.elapsedWallclock() << "ms\n";
		} else {
			nemo::mpi::runWorker(env, world);
		}
	} catch (nemo::exception& e) {
		std::cerr << world.rank() << ":" << e.what() << std::endl;
		env.abort(e.errorNumber());
	} catch (boost::mpi::exception& e) {
		std::cerr << world.rank() << ": " << e.what() << std::endl;
		env.abort(-1);
	}

	return 0;
}



/*! \note when Master is a proper subclass of Simulation, we can share code
 * between the two run functions. */
int
runNoMPI(unsigned ncount, unsigned scount, unsigned duration, const char* filename)
{
	nemo::Network* net = nemo::random::construct(ncount, scount, false);
	nemo::Configuration conf;
	nemo::Simulation* sim = nemo::simulation(*net, conf);

	std::ofstream file(filename);

	sim->resetTimer();
	for(unsigned ms=0; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		file << ms << ": ";
		std::copy(fired.begin(), fired.end(), std::ostream_iterator<unsigned>(file, " "));
		file << std::endl;
	}
	std::cout << "Simulated " << sim->elapsedSimulation() << "ms "
		<< "in " << sim->elapsedWallclock() << "ms\n";
	delete net;

	return 0;
}


int
main(int argc, char* argv[])
{
	if(argc < 4) {
		std::cerr << "usage: example <ncount> <duration> <outfile> [--nompi]\n";
		return -1;
	}

	unsigned ncount = atoi(argv[1]);
	unsigned duration = atoi(argv[2]);
	unsigned scount = 1000;
	char* filename = argv[3];
	bool usingMpi = true;

	if(argc == 5 && strcmp(argv[4], "--nompi") == 0) {
		usingMpi = false;
	}

	if(usingMpi) {
		return run(argc, argv, ncount, scount, duration, filename);
	} else {
		return runNoMPI(ncount, scount, duration, filename);
	}
}
