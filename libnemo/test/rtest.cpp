/* Test that we get the same result as previous runs */

#include <cstring>
#include <iostream>
#include <fstream>

#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <examples.hpp>



void
run(nemo::Network* net, 
	nemo::Configuration conf,
	backend_t backend,
	unsigned seconds,
	const std::string& filename,
	bool creating) // are we comparing against existing data or creating fresh data
{
	std::cerr << "running test\n";
	using namespace std;

	conf.setBackend(backend);

	fstream file;
	//! \todo determine canonical filename based on configuration
	file.open(filename.c_str(), creating ? ios::out : ios::in);
	if(!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		return;
	}

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* nidx;

	unsigned ce, ne; // expexted values

	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {
		sim->step();
		sim->readFiring(&cycles, &nidx);
		for(size_t i = 0; i < cycles->size(); ++i) {
			unsigned c = cycles->at(i);
			unsigned n = nidx->at(i);
			if(creating) {
				file << cycles->at(i) << "\t" << nidx->at(i) << "\n";
			} else {
				BOOST_REQUIRE(!file.eof());
				file >> ce >> ne;
				BOOST_REQUIRE(c == ce);
				BOOST_REQUIRE(n == ne);
			}
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
	bool stdp = false;
	boost::scoped_ptr<nemo::Network> torus(nemo::torus::construct(4, 1000, stdp, 64, false));
	nemo::Configuration conf;
	conf.setFractionalBits(26);

	run(torus.get(), conf, NEMO_BACKEND_CUDA, 4, "test-cuda.dat", creating);
	run(torus.get(), conf, NEMO_BACKEND_CPU, 4, "test-cpu.dat", creating);
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
