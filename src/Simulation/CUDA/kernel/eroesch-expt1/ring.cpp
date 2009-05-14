//! \file ring.cpp

/*! Simple ring-conected network (for debugging)
 *
 * \author Andreas Fidjeland */

#include "izhikevich.h"
#include "Cluster.hpp"
#include "Simulation.hpp"
#include <boost/program_options.hpp>
#include <iostream>
#include <stdio.h>

int g_spikingIndex;


int
processCmdLine(int argc, char** argv,
		int* simCycles,
		int* size,
		int* step,
		int* reportFlags,
		HandlerError(**firingHandler)(FILE*, char*, int, int, uchar),
		FILE** firingFile)
{
	namespace po = boost::program_options;

	try {
		po::options_description desc("Generic options");
		desc.add_options()
			("help", "produce help message")
			("verbose", "print simulation information")
			("cycles", po::value<int>(simCycles)->default_value(1000), 
			 		"number of simulation cycles")
			("size", po::value<int>(size)->default_value(1024),
			 		"number of neurons")
			("step", po::value<int>(step)->default_value(1),
			 		"step size (in neuron indices) between presynaptic and postsynaptic neurons in ring")
			("output-colour", "print firing pattern to terminal (colour)")
			("output-file", po::value<std::string>(), "write firing pattern to file")
			("spike-index", po::value<int>(&g_spikingIndex)->default_value(0),
			 		"neuron which should spike")
			("verify", po::value<std::string>(), "verify firing pattern against file");
			
		po::variables_map vm;
		po::store( po::command_line_parser(argc, argv).options(desc).run(), vm );
		po::notify(vm);

		if(vm.count("help")) {
			std::cout << desc << "\n";
			return 1;
		}

		/* Firing output.
		 * Take care to only create file once */
		if(vm.count("output-file")) {
			*firingHandler = printFiringRaw;
			*firingFile = fopen(vm["output-file"].as<std::string>().c_str(), "w");
		} else if(vm.count("output-colour")) {
			*firingHandler = printFiringTermColour;
			*firingFile = stdout;
		} else if(vm.count("verify")) {
			*firingHandler = verifyFiring;
			*firingFile = fopen(vm["verify"].as<std::string>().c_str(), "r");
		} 

		if(vm.count("verbose")) {
			*reportFlags = REPORT_TIMING | REPORT_FIRING | REPORT_MEMORY;
		} else {
			*reportFlags = 0;
		}
	}

	catch(std::exception& e) {
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch(...) {
		std::cerr << "Exception of unknown type!\n";
		return 1;
	}

	return 0;
}



/* Single spike at beginning of simulation */
void
firingStimulus(char* firing, int n, int /*ms*/)
{
	static bool fired = false;

	memset(firing, 0, n);

	if(!fired) {
		firing[g_spikingIndex] = 1;
		fired = true;
	} 
}



int
main(int argc, char** argv)
{
	int simCycles;
	int size;
	int step;
	int reportFlags;
	HandlerError(*firingHandler)(FILE*, char*, int, int, uchar) = NULL;
	FILE* firingFile = stdout;

	int cmdLineError = processCmdLine(argc, argv,
			&simCycles,
			&size,
			&step,
			&reportFlags,
			&firingHandler,
			&firingFile);
	if(cmdLineError)
		return cmdLineError;

	//! \todo use different sizes here
	//! \todo check command line for handler
	Cluster ring(size);

	for(int pre=0; pre<ring.n; ++pre) {
		ring.setA(pre, 0.02f);
		ring.setB(pre, 0.2f);
		float r = 0.5;
		ring.setC(pre, -65.0f + 15.0f*r*r);
		ring.setD(pre, 8.0f - 6.0f*r*r );
		ring.setV(pre, -65.0f);
		ring.setU(pre, 0.2*-65.0f);
	}

	for(int pre=0; pre<ring.n; ++pre) {
		int post = (pre+step) % ring.n;
		ring.connect(pre, post, 200.0f, 1);
	}
	//! \todo this should be enabled by the simulation, if an external stimulus is non-NULL
	ring.enableExternalFiring();

	Simulation sim;
	sim.addCluster(ring);

	SimStatus status = sim.run(simCycles,
			1, 
			reportFlags,
			1.00f, // scaling factor
			NULL,
			&firingStimulus,
			firingHandler, firingFile,
			NULL, NULL);

	fclose(firingFile);

	return status;
}
