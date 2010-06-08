/* Test that we get the same result as previous runs */

#include <iostream>
#include <fstream>

#include <string.h>

#include <nemo.hpp>
#include <examples.hpp>



/*! \return 0 on success, 1 on failure */
int
run(nemo::Network* net, 
	const nemo::Configuration& conf,
	unsigned seconds,
	const std::string& filename,
	bool creating) // are we comparing against
{
	using namespace std;

	unsigned status = 0;
	fstream file;
	//! \todo determine canonical filename based on configuration
	file.open(filename.c_str(), creating ? ios::out : ios::in);
	if(!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		return 1;
	}

	nemo::Simulation* sim = nemo::Simulation::create(*net, conf);	

	const std::vector<unsigned>* cycles;
	const std::vector<unsigned>* nidx;

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
				unsigned ce, ne;
				//! \todo check for eof here
				file >> ce >> ne;
				if(c != ce || n != ne) {
					std::cerr << "simulation divergence\n"
						<< "\texpected c" << ce << "\tn" << ne << "\n"
						<< "\tfound    c" << c  << "\tn" << n  << std::endl;
					status = 1;
					goto end;
				}
			}
		}
	}

end:
	delete sim;
	file.close();
	return status;
}



int
main(int argc, char* argv[])
{
	bool stdp = false;
	nemo::Network* torus = nemo::torus::construct(4, 1000, stdp, 64, false);
	nemo::Configuration conf;
	std::string filename("test.dat");

	bool creating = argc == 2 && strcmp(argv[1], "create") == 0;
	int status = run(torus, conf, 4, filename, creating);

	delete torus;
	return status;
}
