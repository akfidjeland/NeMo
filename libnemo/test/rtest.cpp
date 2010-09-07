/* Test that we get the same result as previous runs */

#include <cstring>
#include <iostream>
#include <fstream>

#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <examples.hpp>

#include "utils.hpp"



void
run(nemo::Network* net, 
	backend_t backend,
	unsigned seconds,
	const std::string& filename,
	bool stdp,
	bool creating) // are we comparing against existing data or creating fresh data
{
	using namespace std;

	nemo::Configuration conf = configuration(stdp, 1024, backend);
	std::cerr << "running test on " << conf.backendDescription() << " with stdp=" << stdp << std::endl;

	fstream file;
	//! \todo determine canonical filename based on configuration
	file.open(filename.c_str(), creating ? ios::out : ios::in);
	if(!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		return;
	}

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	unsigned ce, ne; // expexted values

	for(unsigned s = 0; s < seconds; ++s) {
		for(unsigned ms = 0; ms < 1000; ++ms) {
			const std::vector<unsigned>& fired = sim->step();
			for(std::vector<unsigned>::const_iterator ni = fired.begin(); ni != fired.end(); ++ni) {
				unsigned c = s * 1000 + ms;
				unsigned n = *ni;
				if(creating) {
					file << c << "\t" << n << "\n";
				} else {
					BOOST_REQUIRE(!file.eof());
					file >> ce >> ne;
					BOOST_REQUIRE(c == ce);
					BOOST_REQUIRE(n == ne);
				}
			}
		}
		if(stdp) {
			sim->applyStdp(1.0);
		}
	}

	if(!creating) {
		/* Read one more word to read off the end of the file. We need to make
		 * sure that we're at the end of the file, as otherwise the test will
		 * pass if the simulation produces no firing */
		file >> ce >> ne;
		BOOST_REQUIRE(file.eof());
	}
}



void runTorus(bool creating)
{
	{
		bool stdp = false;
		boost::scoped_ptr<nemo::Network> torus(nemo::torus::construct(4, 1000, stdp, 64, false));
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda.dat", stdp, creating);
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu.dat", stdp, creating);
	}

	{
		bool stdp = true;
		boost::scoped_ptr<nemo::Network> torus(nemo::torus::construct(4, 1000, stdp, 64, false));
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda-stdp.dat", stdp, creating);
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu-stdp.dat", stdp, creating);
	}
}


void
checkData()
{
	runTorus(false);
}



bool
init_unit_test_suite()
{
	boost::unit_test::test_suite* ts = BOOST_TEST_SUITE("rtest");
	ts->add(BOOST_TEST_CASE(&checkData));
	boost::unit_test::framework::master_test_suite().add(ts);
	return true;
}


int
main(int argc, char* argv[])
{
	bool creating = argc == 2 && strcmp(argv[1], "create") == 0;
	if(creating) {
		runTorus(true);
		std::cerr << "re-generated data";
		return 0;
	} else {
		return ::boost::unit_test::unit_test_main(&init_unit_test_suite, argc, argv);
	}
}
